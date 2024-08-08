import logging
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.functions import col, log, when


def configure_logging() -> None:
    """Configures the logging settings for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def preprocess_data(spark: SparkSession) -> DataFrame:
    """Loads and preprocesses the real estate dataset.

    Adds engineered features such as LogDistanceToMRT, HouseAgeSquared, and Interaction.

    Args:
        spark (SparkSession): The SparkSession object.

    Returns:
        DataFrame: The preprocessed DataFrame.
    """
    logging.info("Loading and preprocessing data.")
    data = (
        spark.read.option("header", "true")
        .option("inferSchema", "true")
        .csv(
            "file:///SparkCourse/Machine Learning with Spark ML/realestate.csv"
        )
    )
    # Handling null values if present
    data = data.na.fill(0)

    data = data.withColumn("LogDistanceToMRT", log(col("DistanceToMRT") + 1))
    data = data.withColumn("HouseAgeSquared", col("HouseAge") ** 2)
    data = data.withColumn(
        "Interaction", col("LogDistanceToMRT") * col("NumberConvenienceStores")
    )
    return data


def build_pipeline() -> Pipeline:
    """Builds the machine learning pipeline for random forest regression.

    The pipeline includes stages for feature vector assembly and standardization.

    Returns:
        Pipeline: The configured machine learning pipeline.
    """
    logging.info("Building the machine learning pipeline.")
    assembler = VectorAssembler(
        inputCols=[
            "HouseAge",
            "LogDistanceToMRT",
            "NumberConvenienceStores",
            "HouseAgeSquared",
            "Interaction",
        ],
        outputCol="features",
    )
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
    rf = RandomForestRegressor(
        featuresCol="scaledFeatures", labelCol="PriceOfUnitArea", seed=42
    )
    return Pipeline(stages=[assembler, scaler, rf])


def tune_model(pipeline: Pipeline, train_df: DataFrame) -> CrossValidator:
    """Tunes the machine learning model using cross-validation.

    Constructs a parameter grid for hyperparameter tuning and evaluates model performance using RMSE.

    Args:
        pipeline (Pipeline): The machine learning pipeline to be tuned.
        train_df (DataFrame): The training dataset.

    Returns:
        CrossValidator: The cross-validated model with the best parameters.
    """
    logging.info("Setting up cross-validation and parameter tuning.")

    rf: RandomForestRegressor = pipeline.getStages()[
        -1
    ]  # Access the RandomForest stage directly
    param_grid = (
        ParamGridBuilder()
        .addGrid(rf.numTrees, [100, 150])
        .addGrid(rf.maxDepth, [10, 20])
        .addGrid(rf.minInstancesPerNode, [1, 2])
        .addGrid(rf.maxBins, [32, 64])
        .build()
    )

    crossval = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=RegressionEvaluator(
            labelCol="PriceOfUnitArea", metricName="rmse"
        ),
        numFolds=5,
        seed=42,
    )
    return crossval.fit(train_df)


def evaluate_model(predictions: DataFrame) -> None:
    """Evaluates the model using various regression metrics.

    Args:
        predictions (DataFrame): The DataFrame containing model predictions.
    """
    logging.info("Evaluating model performance.")
    metrics = ["rmse", "mae", "mse", "r2"]
    for metric in metrics:
        evaluator = RegressionEvaluator(
            labelCol="PriceOfUnitArea",
            predictionCol="prediction",
            metricName=metric,
        )
        value = evaluator.evaluate(predictions)
        logging.info(f"Optimized RandomForest {metric.upper()}: {value:.4f}")


def main() -> None:
    """Main function to execute the real estate modeling script.

    This function sets up Spark, preprocesses data, builds and tunes the model,
    and evaluates the model's performance.
    """
    configure_logging()
    spark = SparkSession.builder.appName(
        "EnhancedRealEstateModeling"
    ).getOrCreate()

    try:
        data = preprocess_data(spark)
        train_df, test_df = data.randomSplit([0.8, 0.2], seed=42)

        pipeline = build_pipeline()
        cv_model = tune_model(pipeline, train_df)
        best_model = cv_model.bestModel
        predictions = best_model.transform(test_df)

        evaluate_model(predictions)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        spark.stop()
        logging.info("Spark session stopped.")


if __name__ == "__main__":
    main()
