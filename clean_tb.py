try:
    from pycrayon import CrayonClient
except ImportError:
    CrayonClient = None
cc = CrayonClient(hostname='127.0.0.1')
cc.remove_all_experiments()