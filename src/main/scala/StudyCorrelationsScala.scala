import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{DataFrame, SparkSession}

object StudyCorrelationsScala {

  def trainRandomForestClassifier(training: DataFrame, test: DataFrame): Unit = {
    val rf = new RandomForestClassifier().setLabelCol("label").setFeaturesCol("features").setNumTrees(10)
    val model = rf.fit(training)

    val predictions = model.transform(test)
    predictions.show(5)

    // Show the metrics
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println(s"Test Error = ${(1.0 - accuracy)}")
  }

  def trainLinearRegression(training: DataFrame, test: DataFrame): Unit = {
    val lr = new LinearRegression().setMaxIter(1000).setRegParam(0.6).setElasticNetParam(0.8)
    val model = lr.fit(training)

    println(s"Coefficients: ${model.coefficients} Intercept: ${model.intercept}")

    val trainingSummary = model.summary
    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
    trainingSummary.residuals.show()
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    println(s"r2: ${trainingSummary.r2}")

    val predictions = model.transform(test)
    predictions.show

    // Show the metrics
    val evaluator = new BinaryClassificationEvaluator().setLabelCol("label").setRawPredictionCol("prediction")
      .setMetricName("areaUnderROC")
    val accuracy = evaluator.evaluate(predictions)
    println(s"Accuracy: $accuracy")
  }

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder
      .master("local[*]")
      .appName("HomeworkSpark")
      .getOrCreate

    spark.sparkContext.setLogLevel("ERROR")

    val file = "normalized2007.csv"
    val inputPath = "src/main/resources/outputLinearRegression/" + file

    val normalized_DF = spark.read.format("csv").option("header", "true").load(inputPath)

    val df = normalized_DF.withColumn("Month", col("Month").cast(DoubleType))
      .withColumn("DayofMonth", col("DayofMonth").cast(DoubleType))
      .withColumn("DayOfWeek", col("DayOfWeek").cast(DoubleType))
      .withColumn("DepTime", col("DepTime").cast(DoubleType))
      .withColumn("FlightNum", col("FlightNum").cast(DoubleType))
      .withColumn("TailNum", col("TailNum").cast(DoubleType))
      .withColumn("CRSDepTime", col("CRSDepTime").cast(DoubleType))
      .withColumn("CRSArrTime", col("CRSArrTime").cast(DoubleType))
      .withColumn("CRSElapsedTime", col("CRSElapsedTime").cast(DoubleType))
      .withColumn("ArrDelay", col("ArrDelay").cast(DoubleType))
      .withColumn("Origin", col("Origin").cast(DoubleType))
      .withColumn("Dest", col("Dest").cast(DoubleType))
      .withColumn("DepDelay", col("DepDelay").cast(DoubleType))
      .withColumn("Distance", col("Distance").cast(DoubleType))
      .withColumn("UniqueCarrier", col("UniqueCarrier").cast(DoubleType))
      .withColumn("TaxiOut", col("TaxiOut").cast(DoubleType))

    var statistic_DF = df.withColumnRenamed("ArrDelay", "label")
    val colNamesList = statistic_DF.columns
    var featuresColsNames = Array[String]()

    for (i <- colNamesList.indices) {
      val colName = colNamesList(i)

      if (colName != "label") {
        val correlation = statistic_DF.stat.corr("label", colNamesList(i))
        println(s"correlation between column ArrDelay and $colName = $correlation")
        if (correlation < 0.29) {
          statistic_DF = statistic_DF.drop(colName)
        } else {
          featuresColsNames = colName +: featuresColsNames
        }
      }
    }

    statistic_DF.printSchema()
    statistic_DF.show(5)

    val assembler = new VectorAssembler().setInputCols(featuresColsNames).setOutputCol("features")
    val model_DF = assembler.transform(statistic_DF)

    model_DF.printSchema()
    model_DF.show(5)

    val Array(training, test) = model_DF.select("label", "features").
      randomSplit(Array(0.7, 0.3), seed = 300000)

    //trainRandomForestClassifier(training, test)
    trainLinearRegression(training, test)
  }
}
