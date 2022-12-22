from pickle import FALSE
import flwr as fl
import numpy as np
from typing import Callable, Dict, Optional, Tuple

if __name__=='__main__':
    fl.server.start_server(server_address="127.0.0.1:8081",config={"num_rounds": 3})

    