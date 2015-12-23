/*
 * Big Data Analytics
 * Fall 2015
 * 
 * Jie Yuan
 * Ziyu He
 * Yubin Shen
 */

import java.io.IOException;
import java.lang.SecurityException;
import java.util.logging.Logger;
import java.util.ArrayList;

import java.util.Scanner;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Random;

import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.fs.FileSystem;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import org.apache.hadoop.mapreduce.Counter;
import org.apache.hadoop.mapreduce.Counters;

/**
 * This class receives an input file and trains a neural network using backpropagation.
 * Each Mapper receives a subset of the original data, and trains a network independently.
 * The reducer then selects the best network based on error rate.
 * A model file with weights is output to a new directory.
 */
public class NeuralNet {

    public static class TrainMapper
            extends Mapper<Object, Text, Text, Text>{

        private static final Logger logger =
                        Logger.getLogger(TrainMapper.class.getName());
        
        private final static IntWritable one = new IntWritable(1);
        private Text word = new Text();

        private static ArrayList<String> heldStrings = new ArrayList<String>();

        public void map(Object key, Text value, Context context)
                throws IOException, InterruptedException {
            

            Configuration conf = context.getConfiguration();
            
            heldStrings.add(value.toString());
            if( heldStrings.size() >= 500 ) {
                // randomize a complete network
                int[] layerNodes = new int[]{784, 400, 4};
                double[][] inputHiddenMatrix = new double[layerNodes[0] + 1][layerNodes[1]];
                double[][] hiddenOutputMatrix = new double[layerNodes[1] + 1][layerNodes[2]];
                for( int i = 0; i < inputHiddenMatrix.length; i++ ) {
                    for( int j = 0; j < inputHiddenMatrix[i].length; j++ ) {
                        Random rand = new Random();
                        inputHiddenMatrix[i][j] = rand.nextDouble() * 2.0 - 1.0;
                    }
                }
                for( int i = 0; i < hiddenOutputMatrix.length; i++ ) {
                    for( int j = 0; j < hiddenOutputMatrix[i].length; j++ ) {
                        Random rand = new Random();
                        hiddenOutputMatrix[i][j] = rand.nextDouble() * 2.0 - 1.0;
                    }
                }
                
                ArrayList<Double> runningError = new ArrayList<Double>();
                int ave_count = 100;
                double globalAveError = 0.0;
                for( int cycle = 0; cycle < 5000; cycle++) {
                    
                    Random rand = new Random();
                    int index = rand.nextInt(heldStrings.size());
                    String[] rowFields = heldStrings.get(index).toString().split(",");
                    int numberClass = Integer.valueOf(rowFields[0]);
                    double[] pixels = new double[rowFields.length];
                    pixels[0] = 1; //bias term
                    for(int i = 1; i < rowFields.length; i++) {
                        pixels[i] = Double.valueOf(rowFields[i]);
                    }
                
                    //forward prop
                    double[] hiddenVals = new double[layerNodes[1] + 1];
                    hiddenVals[0] = 1.0; //bias term
                    for(int j = 1; j < layerNodes[1]; j++) {
                        double input = 0.0;
                        for(int i = 0; i < pixels.length; i++) {
                            input += pixels[i]*inputHiddenMatrix[i][j];
                        }
                        hiddenVals[j] = 1.0/(1.0 + Math.exp(-input));
                    }

                    double[] outputVals = new double[layerNodes[2]];
                    for(int j = 0; j < outputVals.length; j++) {
                        double input = 0.0;
                        for(int i = 0; i < hiddenVals.length; i++) {
                            input += hiddenVals[i]*hiddenOutputMatrix[i][j];
                        }
                        outputVals[j] = 1.0/(1.0 + Math.exp(-input));
                    }
                    
                    //pass values to reducer
                    double maxVal = -1.0;
                    int maxInd = -1;
                    for(int i = 0; i < outputVals.length; i++) {
                        if(outputVals[i] > maxVal) {
                            maxVal = outputVals[i];
                            maxInd = i;
                        }
                    }
                    
                    //back prop
                    //edit the parameters of matrices, write them to file
                    double[] t_k = new double[layerNodes[2]]; //true output
                    t_k[numberClass] = 1.0f;
                    
                    //calculate error
                    double totalErr = 0.0;
                    for( int i = 0; i < outputVals.length; i++ ) {
                        totalErr += (outputVals[i] - t_k[i]) * (outputVals[i] - t_k[i]);
                    }
                    runningError.add(totalErr);
                    if(runningError.size() > ave_count) {
                        runningError.remove(0);
                    }
                    double aveError = 0.0;
                    for( Double err : runningError ) {
                        aveError += err;
                    }
                    aveError = aveError / runningError.size();
                    globalAveError = aveError;
                    
                    if(cycle % ave_count == 0) {
                        //write error to file
                        PrintWriter outFile = new PrintWriter(new FileWriter("nn_mean_errors.txt",true));
                        outFile.println(aveError);
                        outFile.close();
                    }
                    
                    double[] d_k = new double[layerNodes[2]];
                    for(int i = 0; i < d_k.length; i++) {
                        double o_k = outputVals[i];
                        d_k[i] = o_k * (1.0f - o_k) * (t_k[i] - o_k);
                    }
                    double[] d_j = new double[layerNodes[1]];
                    for(int j = 0; j < d_j.length; j++) {
                        double o_j = hiddenVals[j];
                        double sum_k = 0;
                        for(int k = 0; k < d_k.length; k++) {
                            sum_k += d_k[k] * hiddenOutputMatrix[j][k];
                        }
                        d_j[j] = o_j * (1.0f - o_j) * sum_k;
                    }
                    
                    for(int i = 0; i < hiddenOutputMatrix.length; i++) {
                        for(int j = 0; j < hiddenOutputMatrix[i].length; j++) {
                            hiddenOutputMatrix[i][j] += 0.1 * d_k[j] * hiddenVals[i];
                        }
                    }
                    
                    for(int i = 0; i < inputHiddenMatrix.length; i++) {
                        for(int j = 0; j < inputHiddenMatrix[i].length; j++) {
                            inputHiddenMatrix[i][j] += 0.1 * d_j[j] * pixels[i];
                        }
                    }
                }

                //pass weight updates to reducer
                for(int i = 0; i < inputHiddenMatrix.length; i++) {
                    for(int j = 0; j < inputHiddenMatrix[i].length; j++) {
                        context.write(new Text("IH:" + i + ":" + j), new Text("" + globalAveError + ":" + inputHiddenMatrix[i][j]));
                    }
                }
                for(int i = 0; i < hiddenOutputMatrix.length; i++) {
                    for(int j = 0; j < hiddenOutputMatrix[i].length; j++) {
                        context.write(new Text("HO:" + i + ":" + j), new Text("" + globalAveError + ":" + hiddenOutputMatrix[i][j]));
                    }
                }
                heldStrings.clear();
            }
        }
    }

    public static class TrainReducer
            extends Reducer<Text, Text, Text, DoubleWritable> {

        private static final Logger logger =
                        Logger.getLogger(TrainReducer.class.getName());
        
        public void reduce(Text key, Iterable<Text> values,
                                Context context)
                                    throws IOException, InterruptedException {
                                    
            Configuration conf = context.getConfiguration();
            
            double minError = Double.MAX_VALUE;
            double bestWeight = 0;
            for( Text val : values ) {
                String[] fields = val.toString().split(":");
                double aveError = Double.valueOf(fields[0]);
                double weight = Double.valueOf(fields[1]);
                if( aveError < minError )   {
                    minError = aveError;
                    bestWeight = weight;
                }
            }
            context.write(key, new DoubleWritable(bestWeight));
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        
        Job job = Job.getInstance(conf, "word count");
        job.setJarByClass(NeuralNet.class);
        job.setMapperClass(TrainMapper.class);
        job.setReducerClass(TrainReducer.class);
        
        job.setNumReduceTasks(1);
        
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(DoubleWritable.class);
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        Path inputFilePath = new Path(args[0]);
        Path outputFilePath = new Path(args[1]);
        FileSystem fs = FileSystem.newInstance(conf);
        if(fs.exists(outputFilePath)) {
            fs.delete(outputFilePath, true);
        }
        
        job.waitForCompletion(true);
    }
}
