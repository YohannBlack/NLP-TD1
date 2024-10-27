import click
import joblib

from dataset import make_dataset
from features import make_features
from models import make_model, evaluate_model

@click.group()
def cli():
    pass

@click.command()
@click.option("--input_filename", required=True, type=str, default="data/raw/Movies.csv", help="Input filename")
@click.option("--model_name", required=True, type=str, default="DecisionTree", help="Choose model between DecisionTree, RandomForest, GaussianNB")
@click.option("--model_dump_filename", required=True, type=str, default="models/model.json", help="File to dump model")
def train(input_filename, model_name, model_dump_filename):
    df = make_dataset(input_filename)
    X, y = make_features(df)

    model = make_model(model_name)
    
    model.fit(X, y)

    joblib.dump(model, model_dump_filename)
    print(f"Model saved to {model_dump_filename}")

@click.command()
@click.option("--input_filename", required=True, type=str, default="data/raw/Movies.csv", help="Trainin data")
@click.option("--model_dump_filename", required=True, type=str, default="models/model.json", help="Model to load")
@click.option("--output_filename", required=True, type=str, default="data/processed/prediction.csv", help="Output file for predictions")
def predict(input_filename, model_dump_filename, output_filename):
    df = make_dataset(input_filename)
    X, y = make_features(df)

    model = joblib.load(model_dump_filename)
    y_pred = model.predict(X)

    df['prediction'] = y_pred
    df.to_csv(output_filename, index=False)
    print(f"Predictions saved to {output_filename}")

@click.command()
@click.option("--input_filename", required=True, type=str, default="data/raw/Movies.csv", help="Input filename")
@click.option("--model_dump_filename", required=True, type=str, default="models/model.json", help="Model to load")
@click.option("--nb_cross_validation", required=True, type=int, default=5, help="Number of cross validation")
def evaluate(input_filename, model_dump_filename, nb_cross_validation):
    df = make_dataset(input_filename)
    X, y = make_features(df)

    model = joblib.load(model_dump_filename)
    
    evaluate_model(model, X, y, nb_cross_validation)

    


cli.add_command(train)
cli.add_command(predict)
cli.add_command(evaluate)

if __name__ == "__main__":
    cli()