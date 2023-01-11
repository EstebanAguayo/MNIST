# Check that with the given model, when we input x of a given 
# shape it gives back a model with shape Y

import pytest
import torch
from src.models.model import MyAwesomeModel

def test_output_shape():
    model = MyAwesomeModel(784,10) #Hardcoded inputs, not good
    X = torch.rand(1,784)
    assert list(model(X).reshape(1,-1).shape) == [1,10]

def test_error_on_wrong_shape():
    model = MyAwesomeModel(784,10)
    with pytest.raises(ValueError, match='Expected input to a 2D tensor'):    
        model(torch.randn(64,784,3))
    with pytest.raises(ValueError, match='Expected each sample to have shape 784 input values'):
        model(torch.randn(64,74))
