import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{sum, col}
import org.apache.spark.sql.functions.{isnan, when, count, col}

object LoadDataScala {

  def main(args: Array[String]) {
    val spark = SparkSession.builder
      .master("local")
      .appName("HomeworkSpark")
      .getOrCreate

    val path = "src/main/resources/2007.csv"

    val initial_DF = spark.read.format("csv").option("header", "true").load(path)

    val second_DF = initial_DF.drop("ArrTime")
      .drop("ActualElapsedTime")
      .drop("AirTime")
      .drop("TaxiIn")
      .drop("Diverted")
      .drop("CarrierDelay")
      .drop("WeatherDelay")
      .drop("NASDelay")
      .drop("SecurityDelay")
      .drop("LateAircraftDelay")

    // We use this to see NA values.
    // second_DF.select(second_DF.columns.map(c => sum(col(c).isNull.cast("int")).alias(c)): _*).show

    val third_DF = second_DF.drop("CancellationCode")

    // TODO: Look for missing values.
    third_DF.select([count(when(isnan(c), c)).alias(c) for c in df.columns]).show()

    // TODO: Check Correlations between variables

    println(initial_DF.printSchema())
  }
}
