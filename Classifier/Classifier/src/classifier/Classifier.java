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

    private static int pixelSize = 28;
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
    private static int lineCnt = 0;
    private static List<String> trainingData = new ArrayList<>();
    private static List<String> testingData = new ArrayList<>();
    private static boolean Last = true;
    private static int positionRange = 0;
    private static int greyvalueRange = 0;
    /**
     * Parameters
     */
    static boolean newFile = true;
    static boolean minst = false;
    static boolean normalize = true;

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
        datalist.clear();
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
                if (normalize) {
                    if (d > 0) {
                        d = 1;
                    }
                }
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
    public static void naive_bayesian_train(List<String> train) throws IOException {
        totalDigits = 0;
        labelDoc = new HashMap<>();
        for (String labels : classNames) {  //init each class
            labelDoc.put(labels, new ClassRecords(labels));
        }
        SingleRecord sr = null;
        int cnt = 0;
        while (cnt < train.size() && (sr = getR(cnt++, train)) != null) {     //get each single record
            ClassRecords cr = sr.getClassRecord();           //get the class of this single features
            if (cr != null) {
                cr.addOneRecord(sr);                             //add this features to the class
                totalDigits++;
            } else {
                System.out.println("???? line:" + lineCnt);
            }
        }
        System.out.println("Number of training data: " + cnt);
        System.out.println("--------------------");
        for (ClassRecords doc : labelDoc.values()) { //for each class digit
            double prior = (double) doc.getRecordsCount() / totalDigits;
            //System.out.println("doc=" + doc.getClassID() + ", prior=" + prior);
            doc.setPrior(prior);    //set priori (number of class digit / total digit
            doc.constructFeatures();
        }
    }

    public static void naive_bayesian_test(FileReader input, PrintWriter predict, List<String> test) throws IOException {
        SingleRecord sr;
        int correct = 0, incorrect = 0;
        int num = 0;
        while (num < test.size() && (sr = getR(num++, test)) != null) {
            ClassRecords correctClass = sr.getClassRecord();
            if (correctClass != null) {
                ClassRecords predictClass = null;
                double bestScore = -1;
                for (ClassRecords trainDoc : labelDoc.values()) {  //for each class in the training data
                    double score = Math.log(trainDoc.getPrior());   //predict score of that class
                    List<String> feature = sr.getFeatures();    //for the testing record, get all features
                    Map<Integer, Map<String, Double>> featureCounter = trainDoc.getFeatureCounter();
                    for (int i = 0; i < feature.size(); i++) {  //for each feature in the testing record
                        double cond = testMethod(i, feature, featureCounter, trainDoc);
                        score += Math.log(cond);
                    }
                    if (predictClass == null || score > bestScore) {
                        predictClass = trainDoc;
                        bestScore = score;
                    }
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
                    //System.out.println(correctClass.getClassID() + "\t" + predictClass.getClassID());
                } else {
                    System.out.println("null");
                }
            }
        }
        predict.close();
        double res = (double) correct / (incorrect + correct);
        System.out.println("Number of testing data: " + (correct + incorrect));
        System.out.println("Correct=" + correct + ", Incorrect=" + incorrect + ", Accuracy=" + res);
        System.out.println("--------------------");
    }

    public static double testMethod(int index, List<String> feature, Map<Integer, Map<String, Double>> featureCounter, ClassRecords trainDoc) {
        double cond = 0;

        int k = positionRange;
        if (k == 0) {
            if (featureCounter.containsKey(index)) {
                String key2 = String.valueOf(feature.get(index));
                HashMap m = (HashMap) featureCounter.get(index);
                if (m.containsKey(key2)) {
                    double tmp = (double) m.get(key2) / ((double) trainDoc.getRecordsCount());
                    cond += tmp;
                }
            }
        } else {
            int[] aroundIndex = new int[k * 9];
            for (int i = 0; i < k; i++) {
                aroundIndex[(k - 1) * 9 + 0] = index;//center
                aroundIndex[(k - 1) * 9 + 1] = index - 1;//left
                aroundIndex[(k - 1) * 9 + 2] = index + 1;//right;
                aroundIndex[(k - 1) * 9 + 3] = index - pixelSize * k;//up
                aroundIndex[(k - 1) * 9 + 4] = index + pixelSize * k;//down
                aroundIndex[(k - 1) * 9 + 5] = index - (pixelSize * k + 1);//up left
                aroundIndex[(k - 1) * 9 + 6] = index - (pixelSize * k - 1);//up right
                aroundIndex[(k - 1) * 9 + 7] = index + (pixelSize * k - 1);//down left
                aroundIndex[(k - 1) * 9 + 8] = index + (pixelSize * k + 1);//down right
            }
            for (int i = 0; i < k * 9; i++) {
                if (aroundIndex[i] >= 0 && aroundIndex[i] < feature.size()) {
                    int original = Integer.parseInt(feature.get(aroundIndex[i]));
                    for (int j = -greyvalueRange; j <= greyvalueRange; j++) {
                        Integer key1 = aroundIndex[i];
                        if (featureCounter.containsKey(key1)) {
                            String key2 = String.valueOf(original + j);
                            HashMap m = (HashMap) featureCounter.get(key1);
                            if (m.containsKey(key2)) {
                                double tmp = (double) m.get(key2) / ((double) trainDoc.getRecordsCount());
                                cond += tmp;
                            }
                        }
                    }
                }
            }
        }

        return cond;
    }

    public static void main(String[] args) throws IOException {
        filePath = "/Users/Yiting/Documents/Dropbox/Projects/CS491ML/Data/";
        //filePath = "C:\\Users\\yli229\\Dropbox\\Projects\\CS491ML\\Data\\";
        if (minst) {
            originalFile = new File(filePath, "minst.txt");
        } else {
            originalFile = new File(filePath, "digits.ssv");
        }
        System.out.println("Input file: " + originalFile.getName());
        if (minst) {
            Last = false;
            if (normalize) {
                System.out.println("Normalize pixel values");
                positionRange = 0;
                greyvalueRange = 0;
            } else {
                positionRange = 1;
                greyvalueRange = 5;
            }
        } else {
            greyvalueRange = 0;
            positionRange = 0;
        }
        trainFile = new File(filePath, "training.txt");
        testFile = new File(filePath, "test.txt");
        predict = new PrintWriter(new BufferedWriter(new FileWriter("predict.dat", true)));
        classNames = new HashSet<>();
        for (int i = 0; i < 10; i++) {
            classNames.add(String.valueOf(i));
        }
        if (newFile) {
            try {
                FileReader input = new FileReader(originalFile);
                FileWriter train = new FileWriter(trainFile);
                FileWriter test = new FileWriter(testFile);
                split(input, train, test, 0.2);
                System.out.println("split into training and testing...");
            } catch (FileNotFoundException ex) {
                Logger.getLogger(Classifier.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        try {
            FileReader trainInput = new FileReader(trainFile);
            FileReader testInput = new FileReader(testFile);

            getRecord(trainInput, trainingData);
            long beginTraining = System.nanoTime();
            naive_bayesian_train(trainingData);
            long endTraining = System.nanoTime();

            getRecord(testInput, testingData);
            naive_bayesian_test(testInput, predict, testingData);

            naive_bayesian_test(trainInput, predict, trainingData);
            
            System.out.println("Training time = " + (endTraining - beginTraining) / 1000000 + "ms");
        } catch (FileNotFoundException ex) {
            Logger.getLogger(Classifier.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}
