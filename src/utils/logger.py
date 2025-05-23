import logging
import os
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

def setup_logger(name, log_file, level=logging.INFO):
    """
    Setup logger with file and console handlers
    
    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level
        
    Returns:
        Logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

class TensorboardLogger:
    """
    Wrapper for TensorboardX SummaryWriter
    """
    def __init__(self, log_dir):
        """
        Initialize tensorboard logger
        
        Args:
            log_dir: Directory to save tensorboard logs
        """
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        
    def log_scalar(self, tag, value, step):
        """
        Log scalar value
        
        Args:
            tag: Data identifier
            value: Value to log
            step: Global step value
        """
        self.writer.add_scalar(tag, value, step)
        
    def log_histogram(self, tag, values, step):
        """
        Log histogram
        
        Args:
            tag: Data identifier
            values: Values to log
            step: Global step value
        """
        self.writer.add_histogram(tag, values, step)
        
    def log_image(self, tag, img_tensor, step):
        """
        Log image
        
        Args:
            tag: Data identifier
            img_tensor: Image tensor
            step: Global step value
        """
        self.writer.add_image(tag, img_tensor, step)
        
    def log_graph(self, model, input_to_model=None):
        """
        Log model graph
        
        Args:
            model: Model to log
            input_to_model: Input to model for graph visualization
        """
        self.writer.add_graph(model, input_to_model)
        
    def close(self):
        """Close tensorboard writer"""
        self.writer.close()
