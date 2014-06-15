package com.kodcu.sparka;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.rdd.RDD;
import scala.Function0;

/**
 * Created by usta on 01.06.2014.
 */
public class App {

    public static void main(String[] args) {

        SparkConf conf = new SparkConf()
                .setMaster("local")
                .setAppName("Naive Bayes Classifier")
                .set("spark.executor.memory", "1g");

        JavaSparkContext context = new JavaSparkContext(conf);

        RDD<LabeledPoint> trainData = MLUtils.loadLabeledData(context.sc(), "train.data");

        NaiveBayesModel trained = NaiveBayes.train(trainData);


        Vector testData = Vectors.dense(new double[]{3, 1, 1, 0, 0, 0});
        double result = trained.predict(testData);


        System.out.println("Result = " + result);


    }
}
