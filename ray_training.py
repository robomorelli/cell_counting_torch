from pathlib import Path
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from model.resunet import *
from model.utils import EarlyStopping, WeightedLoss, load_data_train_eval
from utils import *
from config import *
import argparse

class ResunetRay(tune.Trainable):

  def setup(self, config):
    #setup function is invoked once training starts
    #setup function is invoked once training starts
    #setup function is invoked once training starts
    #use_cuda = config.get("use_gpu") and torch.cuda.is_available()
    #self.device = torch.device("cuda" if use_cuda else "cpu")
    self.device = torch.device("cuda" if False else "cpu")

    self.batch_size = config['batch_size']
    self.validation_split = config['validation_split']
    self.n_features_start = config['n_features_start']
    self.config = config
    num_workers = 0
    self.train_loader, self.validation_loader = load_data_train_eval(batch_size=self.batch_size,
                                                    num_workers=num_workers, validation_split=self.validation_split,
                                                              shuffle_dataset=True, random_seed=42)
    self.model = c_resunet(arch='c-ResUnet', n_features_start = self.n_features_start, n_out = 1,
                                      pretrained = False, progress= True)


  def step(self):
    self.current_ip()
    val_loss = train_class(self.model, self.device,
                           self.train_loader, self.validation_loader, self.config)
    result = {"loss":val_loss}
    return result

  def save_checkpoint(self, checkpoint_dir):
    print("this is the checkpoint dir {}".format(checkpoint_dir))
    checkpoint_path = os.path.join(checkpoint_dir, self.model_name)
    torch.save(self.model.state_dict(), checkpoint_path)
    return checkpoint_path

  def load_checkpoint(self, checkpoint_path):
    self.model.load_state_dict(torch.load(checkpoint_path))
  # this is currently needed to handle Cori GPU multiple interfaces
  def current_ip(self):
    import socket
    hostname = socket.getfqdn(socket.gethostname())
    self._local_ip = socket.gethostbyname(hostname)
    return self._local_ip


if __name__ == "__main__":
  #parser = argparse.ArgumentParser()
  #parser.add_argument("--address", help="echo the string you use here")
  #parser.add_argument("--password", help="echo the string you use here")
  #args = parser.parse_args()
  #print(args.address, args.password)
  #ray.init(address='auto', _node_ip_address=args.address.split(":")[0], _redis_password=args.password)
  ray.init(address='auto', _node_ip_address="10.141.1.32", _redis_password="5241590000000000")

  config = {
      "n_features_start": tune.choice([16]),
      "epochs": tune.choice([100]),
      "lr": tune.choice([0.003, 0.001, 0.0009]),
      #"act_fun": tune.choice(['relu', 'elu']),
      "batch_size": tune.choice([32]),
      "validation_split": tune.choice([0.2, 0.3, 0.4]),
      "w0": tune.choice([1]),
      "w1": tune.choice([1, 1.5, 2])
  }

    #make a config of different config for each model selected based on the model
    # we want to tune


  save_model_path = Path(ModelResultsRay)

  if not (save_model_path.exists()):
      print('creating path')
      os.makedirs(save_model_path)

  sched = ASHAScheduler(metric="loss", mode="min")
  analysis = tune.run(ResunetRay,
                      scheduler=sched,
                      stop={"training_iteration": 10 ** 16},
                      #resources_per_trial={"cpu": 48, "gpu": 1},
                      resources_per_trial={"cpu": 2},
                      num_samples=1,
                      checkpoint_at_end=True,
                      local_dir=save_model_path.as_posix(),
                      name="c-resunet",
                      config=config)

  print("Best config is:", analysis.get_best_config(metric="loss"))