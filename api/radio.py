"""Validated, bounded radio experiments."""
from fastapi import APIRouter
from ml_lab.radio_simulator import RadioConfig, simulate

router = APIRouter(prefix='/api/radio')


@router.post('/simulate')
def radio_simulate(config: RadioConfig):
    return simulate(config)


@router.get('/models')
def radio_models():
    from ml_lab.radio_channels import MODEL_REGISTRY
    return MODEL_REGISTRY

from api.tec import router as tec_router
router.include_router(tec_router)
