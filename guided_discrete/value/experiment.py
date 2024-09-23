import subprocess
import re 



class Experimenter:
    def __init__(self, lrs, n_layers, n_recurs):
        self.lrs = lrs 
        self.n_layers = n_layers
        self.n_recurs = n_recurs
        self.params = []
        self.eval_loss = []

    def run(self):
        for lr in self.lrs:
            for n_layer in self.n_layers:
                for n_recur in self.n_recurs:
                    try:
                        result = subprocess.run(
                                ["python", "train.py", f"--lr={lr}", f"--n_layer={n_layer}", f"--n_recur={n_recur}", f"--epochs={200}"],
                                capture_output=True,
                                text=True,
                                check=True
                        )
                        loss = self.extract_test_loss(result.stdout)
                        self.params.append(f"--lr={lr} --n_layer={n_layer} --n_recur={n_recur}")
                        self.eval_loss.append(loss)
                    except subprocess.CalledProcessError:
                        print(f"Error running experiment")

        param_loss = zip(self.params, self.eval_loss)
        with open("experiment_logs", "a+") as file:
            for param, loss in param_loss:
                file.write(f"{param} loss: {loss}\n")

    def extract_test_loss(self, output_str):
        pattern = r"Test loss: (\d+\.\d+)"
        match = re.search(pattern, output_str)
        if match:
            accuracy = float(match.group(1))
            return accuracy
        else:
            raise ValueError()

                


        
        




if __name__ == "__main__":
    #lrs = [1e-3, 6e-3]
    lrs = [1e-4, 6e-4, 1e-3, 6e-3, 1e-2]
    #n_layers = [1]
    n_layers = [1,2,4, 16, 32]
    #n_recurs = [1]
    n_recurs = [1,2,4,16, 32]
    exp = Experimenter(lrs, n_layers, n_recurs)
    exp.run()