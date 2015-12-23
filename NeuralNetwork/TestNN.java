/*
 * Big Data Analytics
 * Fall 2015
 * 
 * Jie Yuan
 * Ziyu He
 * Yubin Shen
 */

import java.io.IOException;
import java.util.ArrayList;

import java.util.Scanner;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Random;

/**
 * This class reads in a pre-trained model and calculates the output for a given set of
 * input. The highest predicted class is output along with the true class
 */
public class TestNN {

    public static void main(String[] args) throws Exception {
        
        // load the model from a file
        int[] layerNodes = new int[]{784, 400, 4};
        double[][] inputHiddenMatrix = new double[layerNodes[0] + 1][layerNodes[1]];
        double[][] hiddenOutputMatrix = new double[layerNodes[1] + 1][layerNodes[2]];
        Scanner in = new Scanner(new FileReader("nn_model"));
        while(in.hasNextLine()) {
            String line = in.nextLine();
            String[] keyVal = line.split("\t");
            String[] weightID = keyVal[0].split(":");
            double weight = Double.valueOf(keyVal[1]);
            int i = Integer.valueOf(weightID[1]);
            int j = Integer.valueOf(weightID[2]);
            
            if( weightID[0].equals("IH") ) {
                inputHiddenMatrix[i][j] = weight;
            } else {
                hiddenOutputMatrix[i][j] = weight;
            }
        }
        in.close();
        
        // read data from file
        ArrayList<String> lines = new ArrayList<String>();
        Scanner in2 = new Scanner(new FileReader("mnist_test_0123only.csv"));
        while(in2.hasNextLine()) {
            lines.add(in2.nextLine());
        }
        in2.close();
        
        //run feed forward to get class assignment for each point
        PrintWriter writer = new PrintWriter("subset_output.csv", "UTF-8");
        for( String str : lines ) {
            String[] rowFields = str.split(",");
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
            
            // get observed and expected classes
            double maxVal = -1.0;
            int maxInd = -1;
            for(int i = 0; i < outputVals.length; i++) {
                if(outputVals[i] > maxVal) {
                    maxVal = outputVals[i];
                    maxInd = i;
                }
            }
            writer.println(numberClass + "," + maxInd);
        }
        writer.close();
    }
}
