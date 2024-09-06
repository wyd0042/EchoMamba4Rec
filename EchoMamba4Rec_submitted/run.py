import sys
import time
import torch 
import logging
from logging import getLogger
from recbole.utils import init_logger, init_seed
from recbole.trainer import Trainer

from model import EchoMamba4Rec

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.transform import construct_transform
from recbole.utils import (
    init_logger,
    get_model,
    get_trainer,
    init_seed,
    set_color,
    get_flops,
    get_environment,
)

if __name__ == '__main__':

    config = Config(model=EchoMamba4Rec, config_file_list=['config.yaml'])
    init_seed(config['seed'], config['reproducibility'])
    
    
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    
    dataset = create_dataset(config)
    logger.info(dataset)

    
    train_data, valid_data, test_data = data_preparation(config, dataset)

    
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = EchoMamba4Rec(config, train_data.dataset).to(config['device'])
    logger.info(model)
    
    transform = construct_transform(config)
    flops = get_flops(model, dataset, config["device"], logger, transform)
    logger.info(set_color("FLOPs", "blue") + f": {flops}")

    
    trainer = Trainer(config, model)

    
    start_training_time = time.time()
    
    
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, show_progress=config["show_progress"]
    )
    
    end_training_time = time.time()
    training_time = end_training_time - start_training_time
    logger.info(set_color("Training time", "green") + f": {training_time:.2f} seconds")

    # model evaluation
    test_result = trainer.evaluate(
        test_data, show_progress=config["show_progress"]
    )
    
    environment_tb = get_environment(config)
    logger.info(
        "The running environment of this training is as follows:\n"
        + environment_tb.draw()
    )

    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")

    model.eval()  

    with torch.no_grad():  
        for batch in test_data:
            interaction = batch[0] 
            item_seq = interaction['item_id_list'].to(config['device'])  
            item_seq_len = interaction['item_length'].to(config['device']) 
            break 

        
        start_time = time.time()
        _ = model.forward(item_seq, item_seq_len)
        end_time = time.time()

        inference_time = end_time - start_time
        logger.info(set_color("Inference time", "green") + f": {inference_time:.6f} seconds for a batch of size {len(item_seq)}")
