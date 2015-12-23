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
import java.util.Random;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.fs.FileSystem;

import org.apache.hadoop.mapreduce.Counter;
import org.apache.hadoop.mapreduce.Counters;

/**
 * This MapReduce job rearranges the lines of a file in random order
 * and writes the output to a new directory
 */
public class Scramble {

    public static class ScrambleMapper
            extends Mapper<Object, Text, DoubleWritable, Text>{

        public void map(Object key, Text value, Context context)
                throws IOException, InterruptedException {

            Configuration conf = context.getConfiguration();
            Random rand = new Random();
            double order = rand.nextDouble();
                    
            context.write(new DoubleWritable(order), value);
        }
    }

    public static class emptyReducer
            extends Reducer<DoubleWritable, Text, NullWritable, Text> {

        private static final Logger logger =
                        Logger.getLogger(emptyReducer.class.getName());

        public void reduce(DoubleWritable key, Iterable<Text> values,
                                Context context)
                                    throws IOException, InterruptedException {
            
            for (Text val : values) {
                context.write(NullWritable.get(), val);
            }
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "word count");
        job.setJarByClass(Scramble.class);
        job.setMapperClass(ScrambleMapper.class);
        //job.setCombinerClass(emptyReducer.class);
        job.setReducerClass(emptyReducer.class);
        
        job.setMapOutputKeyClass(DoubleWritable.class);
        job.setMapOutputValueClass(Text.class);
        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(Text.class);
        
        Path inputFilePath = new Path(args[0]);
        Path outputFilePath = new Path(args[1]);
        FileInputFormat.addInputPath(job, inputFilePath);
        FileOutputFormat.setOutputPath(job, outputFilePath);
        
        FileSystem fs = FileSystem.newInstance(conf);
        if(fs.exists(outputFilePath)) {
            fs.delete(outputFilePath, true);
        }

        job.waitForCompletion(true);
        
    }
}
