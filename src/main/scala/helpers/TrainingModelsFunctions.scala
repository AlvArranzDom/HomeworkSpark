package helpers

import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor, LinearRegression, LinearRegressionModel, RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.sql.DataFrame
/**
 * Object that has all the functions related to create, fit & transforming ML models.
 */
object TrainingModelsFunctions {

  /**
   * Train the model specified by the entry params with the entry dataset, print the metrics and return the resulting
   * model.
   *
   * @param modelName    model to be trained. Possible Values:
   *                 - "lr" for LinearRegression
   *                 - "rf" for RandomForest
   * @param model_DF     entry dataset for training and testing
   * @param split_values values to split the entry dataset in training and test datasets (Default: (0.7, 0.3)
   * @return result model
   */
  def trainModel(modelName: String, model_DF: DataFrame, split_values: Array[Double] = Array(0.7, 0.3)): Any = {
    val Array(training, test) = model_DF.select("label", "features").
      randomSplit(split_values)

    if (modelName == "lr") {
      trainLinearRegression(training, test)
    } else if (modelName == "rfc") {
      trainRandomForestClassifier(training, test)
    } else if (modelName == "gbtr") {
      trainGBTreeRegression(training, test)
    } else if (modelName == "rfr") {
      trainRandomForestRegression(training, test)
    } else {
      println(s"Model don't implemented to be trained. \n Please use 'lr' to train ")
    }
  }

  /**
   * Train a Random Forest Classifier Model.
   *
   * @param training entry dataset for train the model
   * @param test     entry dataset for test the model
   * @return result model
   */
  private def trainRandomForestClassifier(training: DataFrame, test: DataFrame): RandomForestClassificationModel = {
    val rf = new RandomForestClassifier().setLabelCol("label").setFeaturesCol("features").setNumTrees(10)
    val model = rf.fit(training)

    val predictions = model.transform(test)

    // Show the prediction metrics
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println(s"Test Error = ${(1.0 - accuracy)}")

    model
  }

  /**
   * Train a Linear Regression Model.
   *
   * @param training entry dataset for train the model
   * @param test     entry dataset for test the model
   * @return result model
   */
  private def trainLinearRegression(training: DataFrame, test: DataFrame): LinearRegressionModel = {
    val lr = new LinearRegression().setMaxIter(5).setRegParam(0.8).setElasticNetParam(0)
    val model = lr.fit(training)

    println(s"Coefficients: ${model.coefficients} Intercept: ${model.intercept}")

    // Show the model metrics
    val trainingSummary = model.summary
    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
    trainingSummary.residuals.show()
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    println(s"r2: ${trainingSummary.r2}")

    val predictions = model.transform(test)

    // Show the prediction metrics
    val evaluator = new BinaryClassificationEvaluator().setLabelCol("label").setRawPredictionCol("prediction")
      .setMetricName("areaUnderROC")
    val accuracy = evaluator.evaluate(predictions)
    println(s"Accuracy: $accuracy")

    model
  }
  /**
   * Train a Gradient-boosted tree regression Model.
   *
   * @param training entry dataset for train the model
   * @param test     entry dataset for test the model
   * @return result model
   */
  private def trainGBTreeRegression(training: DataFrame, test: DataFrame): GBTRegressionModel = {
    val gb = new GBTRegressor()
      .setMaxIter(30)
      .setLabelCol("label")
      .setFeaturesCol("features")
    val model = gb.fit(training)

    val predictions = model.transform(test)
    predictions.select("prediction", "label", "features").show(5)

    val evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    println("evaluator successfully created")
    val rmse = evaluator.evaluate(predictions)
    println(s"Root Mean Squared Error (RMSE) on test data $rmse")

    model
  }

  private def trainRandomForestRegression(training: DataFrame, test: DataFrame): RandomForestRegressionModel = {
    val rf = new RandomForestRegressor()
      .setLabelCol("label")
      .setFeaturesCol("features")
    val model = rf.fit(training)

    val predictions = model.transform(test)
    predictions.select("prediction", "label", "features").show(5)

    val evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    println("evaluator successfully created")
    val rmse = evaluator.evaluate(predictions)
    println(s"Root Mean Squared Error (RMSE) on test data $rmse")

    model
  }
  }
