/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package classifier;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author Yiting
 */
public class Classifier {

    //private int pixelSize = 48;
    private static String filePath;
    private static File originalFile;
    private static File trainFile;
    private static File testFile;
    private static PrintWriter predict;
    private static int totalDigits;
    private static Set<String> classNames;
    private static Map<String, ClassRecords> labelDoc;
    private static List<String> correctRes = new ArrayList<>();
    private static List<String> predictRes = new ArrayList<>();
    private static boolean check = true;
    private static int lineCnt = 0;
    private static List<String> trainingData = new ArrayList<>();
    private static List<String> testingData = new ArrayList<>();
    private static boolean Last = true;
    private static int positionRange = 6;

    private static void split(FileReader input, FileWriter train, FileWriter test, double testRatio) throws IOException {
        BufferedReader reader = new BufferedReader(input);
        FileWriter output;
        while (true) {
            if (Math.random() < testRatio) {
                output = test;
            } else {
                output = train;
            }
            String line;
            if ((line = reader.readLine()) != null) {
                output.write(line + "\n");
            } else {
                break;
            }
        }
        input.close();
        train.close();
        test.close();
    }

    public static void getRecord(FileReader input, List<String> datalist) throws IOException {
        lineCnt++;
        SingleRecord sd = null;
        BufferedReader reader = new BufferedReader(input);
        String line;
        while ((line = reader.readLine()) != null) {
            datalist.add(line);
        }
        input.close();
    }

    public static SingleRecord getR(int num, List<String> datalist) {
        SingleRecord sd = null;
        String line = datalist.get(num);
        sd = new SingleRecord();
        String[] digits = line.split("\\s+"); //split by space
        int len = digits.length;
        int start = 0, end = len - 1, labelIndex = len - 1;
        if (!Last) {
            start = 1;
            end = len;
            labelIndex = 0;
        }
        for (; start < end; start++) {
            if (!"".equals(digits[start])) {
                int d = Integer.parseInt(digits[start]);

                if (d > 0) {
                    d = 1;
                }
//                if (d == 0) {
//                    d = 0;
//                } else if (d <= 55) {
//                    d = 1;
//                } else if (d <= 110) {
//                    d = 2;
//                } else if (d <= 165) {
//                    d = 3;
//                } else {
//                    d = 4;
//                }
                sd.addFeature(String.valueOf(d));
            }
        }
        String label = digits[labelIndex];
        if (!labelDoc.containsKey(label)) {
            System.out.println(line);
        }
        sd.setClassRecord(labelDoc.get(digits[labelIndex]));
        return sd;
    }

    /**
     * P(Y|x1, x2, x3, ... ,xn) --> P(x1, x2, x3, ... , xn | Y)*P(Y) -->
     * log(x1|Y) + log(x2|Y) + ... + P(xn|Y) + logP(Y)
     *
     * @throws IOException
     */
    public static void naive_bayesian_train() throws IOException {
        totalDigits = 0;
        labelDoc = new HashMap<>();
        for (String labels : classNames) {  //init each class
            labelDoc.put(labels, new ClassRecords(labels));
        }
        SingleRecord sr = null;
        int cnt = 0;
        while (cnt < trainingData.size() && (sr = getR(cnt++, trainingData)) != null) {     //get each single record
            ClassRecords cr = sr.getClassRecord();           //get the class of this single features
            if (cr != null) {
                cr.addOneRecord(sr);                             //add this features to the class
                totalDigits++;
            } else {
                System.out.println("???? line:" + lineCnt);
            }
        }
        for (ClassRecords doc : labelDoc.values()) { //for each class digit
            double prior = (double) doc.getRecordsCount() / totalDigits;
            System.out.println("doc=" + doc.getClassID() + ", prior=" + prior);
            doc.setPrior(prior);    //set priori (number of class digit / total digit
            doc.constructFeatures();
        }
    }

    public static void naive_bayesian_test(FileReader input, PrintWriter predict) throws IOException {
        SingleRecord sr;
        int correct = 0, incorrect = 0;
        int num = 0;
        while (num < testingData.size() && (sr = getR(num++, testingData)) != null) {
            ClassRecords correctClass = sr.getClassRecord();
            if (correctClass != null) {
                ClassRecords predictClass = null;
                double bestScore = -1;
                for (ClassRecords trainDoc : labelDoc.values()) {  //for each class in the training data
                    double score = Math.log(trainDoc.getPrior());   //predict score of that class
                    List<String> feature = sr.getFeatures();    //for the testing record, get all features
                    Map<String, Integer> featureCounter = trainDoc.getFeatureCounter();
                    Map<Integer, Double> featureDistance = trainDoc.getFeatureDistance();
                    for (int i = 0; i < feature.size(); i++) {  //for each feature in the testing record
                        double cond = testMethod(i, feature, featureCounter, trainDoc);
                        score += Math.log(cond);
                    }
                    if (predictClass == null || score > bestScore) {
                        predictClass = trainDoc;
                        bestScore = score;
                    }
                    //System.out.println("train class=" + trainDoc.getClassID() + ", score=" + score);
                }


                if (predictClass != null) {
                    predict.write(predictClass.getClassID() + "\n");
                    if (predictClass == correctClass) {
                        correct++;
                    } else {
                        incorrect++;
                    }
                    correctRes.add(correctClass.getClassID());
                    predictRes.add(predictClass.getClassID());
                    System.out.println(correctClass.getClassID() + "\t" + predictClass.getClassID());
                } else {
                    System.out.println("null");
                }
            }
        }
        predict.close();
        double res = (double) correct / (incorrect + correct);
        System.out.println("Number of testing data: " + (correct + incorrect));
        System.out.println("Correct=" + correct + ", Incorrect=" + incorrect + ", Accuracy=" + res);
        System.out.println("");
    }

    public static double testMethod(int index, List<String> feature, Map<String, Integer> featureCounter, ClassRecords trainDoc) {
        //int valueRange = 100;
        double cond = 0;
        String[] possibleKeys = new String[positionRange];    //10
        for (int i = 0; i < positionRange; i++) {   //i from index-5 to index + 5
            int simindex = index - positionRange / 2 + i;
            if (simindex >= 0 && simindex < feature.size()) {
                possibleKeys[i] = String.valueOf(simindex) + "," + feature.get(simindex);
                if (featureCounter.containsKey(possibleKeys[i])) {
                    cond += (double) featureCounter.get(possibleKeys[i]) / ((double) trainDoc.getRecordsCount());
                }
            }
        }
//        for (int i = 1; i <= valueRange; i++) {
//            String key = String.valueOf(index) + "," + feature.get(index) + i;
//            if (featureCounter.containsKey(key)) {
//                cond += (double) featureCounter.get(key) / ((double) trainDoc.getRecordsCount());
//            }
//        }
        return cond;
    }

    public static void main(String[] args) throws IOException {
        filePath = "/Users/Yiting/Google Drive/00_Study/CS491ML/project/NB/";
        //filePath = "C:\\Users\\yli229\\Documents\\ML491\\NB";
        originalFile = new File(filePath, "minst");
        //originalFile = new File(filePath, "digits.ssv");
        System.out.println("Input file: " + originalFile.getName());
        if (originalFile.getName().equals("minst")) {
            Last = false;
            positionRange = 1;
        } else {
            positionRange = 1;
        }
        trainFile = new File(filePath, "training.txt");
        testFile = new File(filePath, "test.txt");
        predict = new PrintWriter(new BufferedWriter(new FileWriter("predict.dat", true)));
        classNames = new HashSet<>();
        for (int i = 0; i < 10; i++) {
            classNames.add(String.valueOf(i));
        }
        try {
            FileReader input = new FileReader(originalFile);
            FileWriter train = new FileWriter(trainFile);
            FileWriter test = new FileWriter(testFile);
            split(input, train, test, 0.1);
            System.out.println("split done");
        } catch (FileNotFoundException ex) {
            Logger.getLogger(Classifier.class.getName()).log(Level.SEVERE, null, ex);
        }
        try {
            FileReader trainInput = new FileReader(trainFile);
            FileReader testInput = new FileReader(testFile);
            getRecord(trainInput, trainingData);
            getRecord(testInput, testingData);
            naive_bayesian_train();
            naive_bayesian_test(testInput, predict);
        } catch (FileNotFoundException ex) {
            Logger.getLogger(Classifier.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}
