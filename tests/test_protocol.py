import os
import pytest
import torch
from models import get_model
from zkp_utils import prepare_circuit_inputs
from scripts.rerun_committed import loss_sum
import numpy as np

def test_har_model_accepts_official_feature_vectors():
    assert get_model('har')(torch.zeros(2,561)).shape == (2,6)

def test_precision_changes_after_module_import(monkeypatch):
    monkeypatch.setenv('ZKP_PRECISION','3')
    _,pub,priv=prepare_circuit_inputs(.125,.5,'ABC','a'*64)
    assert priv['calculated_loss']=='125' and pub['threshold']=='500'
    monkeypatch.setenv('ZKP_PRECISION','6')
    _,pub,priv=prepare_circuit_inputs(.125,.5,'ABC','a'*64)
    assert priv['calculated_loss']=='125000' and pub['threshold']=='500000'

@pytest.mark.parametrize('value',[-1,float('nan'),float('inf'),2**64])
def test_invalid_loss_rejected(value):
    public,_,_=prepare_circuit_inputs(value,.5,'ABC','a'*64)
    assert public is None

def test_integer_loss_has_exact_boundary():
    x=np.array([[1,-2]],dtype=np.int64)
    w=np.array([[3,4]],dtype=np.int64)
    y=np.array([[1]],dtype=np.int64)
    assert loss_sum(x,y,w,np.array([5]))==100000000
