import toml
import pandas as pd

CONFIG_PATH = r"C:\Users\james\PycharmProjects\recommendation-engine\config.toml"

config = toml.load(CONFIG_PATH)

print(config)

movies_path = config['paths']['movies']
movies = pd.read_csv(movies_path)
