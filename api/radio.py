"""Validated, bounded radio experiments."""
from fastapi import APIRouter, HTTPException
from ml_lab.radio_simulator import RadioConfig, simulate
from ml_lab.channel_lab import ChannelLabConfig, run_channel_lab
from ml_lab.channel_setup import ChannelSetupRequest, prepare_channel_setup
from ml_lab.radio_daily import DailyRequest, compare_daily
from ml_lab.radio_research import ResearchRequest, ResponseRequest, analyze_research, ionosphere_response

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



@router.post('/daily')
def radio_daily(request: DailyRequest):
    try:
        return compare_daily(request)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc



@router.post('/research')
def radio_research(request: ResearchRequest):
    try:
        return analyze_research(request)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc



@router.post('/response')
def radio_response(request: ResponseRequest):
    try:
        return ionosphere_response(request)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post('/channel-lab')
def channel_lab(request: ChannelLabConfig):
    return run_channel_lab(request)


@router.post('/channel-setup')
def channel_setup(request: ChannelSetupRequest):
    try:
        return prepare_channel_setup(request)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
