import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.DoubleType

object StudyCorrelationsScala {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder
      .master("local[*]")
      .appName("HomeworkSpark")
      .getOrCreate

    spark.sparkContext.setLogLevel("ERROR")

    val file = "normalized1996.csv"
    val inputPath = "src/main/resources/output/" + file

    val normalized_DF = spark.read.format("csv").option("header", "true").load(inputPath)

    val df = normalized_DF.withColumn("Month", col("Month").cast(DoubleType))
      .withColumn("DayofMonth", col("DayofMonth").cast(DoubleType))
      .withColumn("DayOfWeek", col("DayOfWeek").cast(DoubleType))
      .withColumn("DepTime", col("DepTime").cast(DoubleType))
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

    val correlation1 = df.stat.corr("ArrDelay", "Month")
    println(s"correlation between column ArrDelay and Month = $correlation1")

    val correlation2 = df.stat.corr("ArrDelay", "DayofMonth")
    println(s"correlation between column ArrDelay and DayofMonth = $correlation2")

    val correlation3 = df.stat.corr("ArrDelay", "DayOfWeek")
    println(s"correlation between column ArrDelay and DayOfWeek = $correlation3")

    val correlation4 = df.stat.corr("ArrDelay", "DepTime")
    println(s"correlation between column ArrDelay and DepTime = $correlation4")

    val correlation5 = df.stat.corr("ArrDelay", "CRSDepTime")
    println(s"correlation between column ArrDelay and CRSDepTime = $correlation5")

    val correlation6 = df.stat.corr("ArrDelay", "CRSArrTime")
    println(s"correlation between column ArrDelay and CRSArrTime = $correlation6")

    val correlation7 = df.stat.corr("ArrDelay", "UniqueCarrier")
    println(s"correlation between column ArrDelay and UniqueCarrier = $correlation7")

    val correlation8 = df.stat.corr("ArrDelay", "CRSElapsedTime")
    println(s"correlation between column ArrDelay and CRSElapsedTime = $correlation8")

    val correlation9 = df.stat.corr("ArrDelay", "DepDelay")
    println(s"correlation between column ArrDelay and DepDelay = $correlation9")

    val correlation10 = df.stat.corr("ArrDelay", "Origin")
    println(s"correlation between column ArrDelay and Origin = $correlation10")

    val correlation11 = df.stat.corr("ArrDelay", "Dest")
    println(s"correlation between column ArrDelay and Dest = $correlation11")

    val correlation12 = df.stat.corr("ArrDelay", "Distance")
    println(s"correlation between column ArrDelay and Distance = $correlation12")

    val correlation13 = df.stat.corr("ArrDelay", "TaxiOut")
    println(s"correlation between column ArrDelay and TaxiOut = $correlation13")

    // Drop correlation values less than 0.10
    val statistic_DF = df.drop("Month")
      .drop("DayofMonth")
      .drop("DayOfWeek")
      .drop("DepTime")
      .drop("UniqueCarrier")
      .drop("CRSElapsedTime")
      .drop("Origin")
      .drop("Dest")
      .drop("Distance")

    //val Row(coeff1: Matrix) = Correlation.corr(df, "ArrDelay").head
    //println("Pearson correlation matrix:\n" + coeff1.toString)
  }
}
