from model_v2.v1_0.runner import Runner
from ga_classes.envs import EnvGA


if __name__ == "__main__":
    r = Runner(EnvGA, test_set=20, real=True)
    r.run()
    exit()
