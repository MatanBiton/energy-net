"""Corrected EN-05 cases remain separate from EN-02 legacy cases."""
import numpy as np
import pandas as pd
import pytest
from energy_net.gym_envs.pcs_env import PCSEnv
from energy_net.grid_entities.management.price_curve import PriceCurveStrategy

class Tariffs(PriceCurveStrategy):
    use_asymmetric_pricing = True
    def __init__(self, purchase=.1, sale=.05):
        self.purchase, self.sale = purchase, sale
    def calculate_price(self, observation=None):
        return np.broadcast_to(self.purchase, (48,)).copy()
    def calculate_sell_price(self, observation=None):
        return self.calculate_price(observation)
    def calculate_buy_price(self, observation=None):
        return np.broadcast_to(self.sale, (48,)).copy()

@pytest.fixture
def factory(tmp_path):
    def create(load=3., tariffs=None, **kwargs):
        dates = pd.date_range('2025-07-09', periods=144, freq='30min')
        demand, predictions = tmp_path/'demand.csv', tmp_path/'predictions.csv'
        pd.DataFrame({'Datetime': dates, 'Consumption': np.broadcast_to(load, (144,))}).to_csv(demand, index=False)
        pd.DataFrame({'timestamp': dates, 'predicted_consumption': np.ones(144)*3}).to_csv(predictions, index=False)
        return PCSEnv(test_data_file=str(demand), predictions_file=str(predictions),
                      data_mode='corrected', settlement_mode='corrected', verbosity=0,
                      price_strategy=tariffs or Tariffs(), **kwargs)
    return create

@pytest.mark.parametrize('s,u,l,q,b,sn,d,g,r,sale', [
    (0,3,5,3,3,0,2,5,-.5,.1),
    (5,0,3.19168,0,3.19168,1.80832,0,0,0,.1),
    (3,0,3,0,3,0,0,0,0,.1),
    (98,10,3,2,3,97,0,2,-.2,.1),
    (5,-3,5,-3,2,0,3,0,0,.05),
    (5,-3,0,-3,0,2,0,-3,.15,.05),
    (2,-10,0,-2,0,0,0,-2,.1,.05),
    (100,10,0,0,0,100,0,0,0,.05),
    (0,-10,0,0,0,0,0,0,0,.05),
    (5,-3,10,-3,2,0,8,5,-.5,.05),
])
def test_examples(factory,s,u,l,q,b,sn,d,g,r,sale):
    env=factory(l,Tariffs(.1,sale),shortage_penalty=999)
    obs,_=env.reset(start_date=env.test_start_date)
    assert obs[2:4] == pytest.approx((.1,sale))
    env.pcs.storage_units[0].reset(initial_level=s)
    _,actual,_,_,i=env.step([u])
    for key,value in [('realized_action_units',q),('battery_service_units',b),
                      ('storage_after_units',sn),('residual_demand_units',d),
                      ('grid_exchange_units',g),('step_money',r)]:
        assert i[key] == pytest.approx(value)
    assert actual == pytest.approx(r)
    assert i['unserved_energy_units'] == i['step_penalty'] == i['reserve_penalty'] == 0
    assert not i['shortage']
    assert env.shortage_count == env.total_shortage_penalty == 0
    assert env.get_money() == pytest.approx(r)

def test_randomized_reconciliation_and_reset(factory):
    rng=np.random.default_rng(20261116)
    env=factory(rng.uniform(0,15,144))
    for day in range(2):
        env.reset(start_date=env.test_start_date+pd.Timedelta(days=day))
        assert env.get_money() == env.total_reward == 0
        cost=0.
        for k in range(48):
            _,r,done,truncated,i=env.step(rng.uniform(-10,10,1))
            assert 0 <= i['storage_after_units'] <= 100
            assert i['consumption_units'] == pytest.approx(i['battery_service_units']+i['residual_demand_units'])
            assert i['grid_exchange_units'] == pytest.approx(i['consumption_units']+i['storage_after_units']-i['storage_before_units'])
            assert i['residual_demand_units'] >= 0
            assert i['grid_import_units']*i['grid_export_units'] == 0
            expected=.1*i['grid_import_units']-.05*i['grid_export_units']
            assert i['net_electricity_cost'] == pytest.approx(expected)
            assert i['counterparty_receipt']+r == pytest.approx(0)
            cost+=expected
            assert env.get_money() == pytest.approx(-cost)
            assert env.total_reward == pytest.approx(-cost)
            assert done == (k == 47)
            assert not truncated
        with pytest.raises(ValueError,match='ended'):
            env.step([0])

def test_calendar_prices(factory):
    env=factory(0,Tariffs(np.arange(48)/100,np.arange(48)/200))
    obs,_=env.reset(start_date=env.test_start_date+pd.Timedelta(hours=12))
    for k in range(48):
        slot=(24+k)%48
        assert obs[2:4] == pytest.approx((slot/100,slot/200))
        obs,_,_,_,i=env.step([0])
        assert (i['purchase_price'],i['sale_price']) == pytest.approx((slot/100,slot/200))
    assert obs.shape == env.observation_space.shape

@pytest.mark.parametrize('buy,sell',[(-.1,0),(.1,-.1),(.1,.2),(np.nan,0),(np.inf,0),(11,0)])
def test_invalid_tariffs(factory,buy,sell):
    with pytest.raises(ValueError,match='finite|Tariffs'):
        env=factory(0,Tariffs(buy,sell))
        env.reset(start_date=env.test_start_date)

@pytest.mark.parametrize('action',[[np.nan],[np.inf],[11],[-11],[],[0,0]])
def test_invalid_actions_atomic(factory,action):
    env=factory()
    env.reset(start_date=env.test_start_date)
    with pytest.raises(ValueError,match='Action'):
        env.step(action)
    assert env.current_step == env.get_money() == env.pcs.get_total_storage() == 0

def test_strategy_change(factory):
    env=factory()
    env.reset(start_date=env.test_start_date)
    env.step([3])
    env.set_price_strategy(Tariffs(.2,.1))
    obs=env._get_obs()  # refresh before the next decision
    assert obs[2:4] == pytest.approx((.2,.1))
    _,r,_,_,i=env.step([0])
    assert r == pytest.approx(-.6)
    env.reset(start_date=env.test_start_date)
    assert env.get_money() == env.total_reward == env.total_shortage_penalty == 0
