"""Clarke statistical channel."""
from .clarke import clarke

MODEL_REGISTRY = {
    'tec': {'name':'A · ПЭС / холодная плазма','status':'implemented','role':'physical_first_order'},
    'combined': {'name':'A × B · ПЭС + Кларк','status':'implemented','role':'hybrid_research'},
    'clarke': {'name': 'Кларк', 'status': 'implemented', 'role': 'statistical_reference',
               'source': 'https://doi.org/10.1002/j.1538-7305.1968.tb00069.x'},
}
