/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package classifier;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 *
 * @author Yiting
 */
public class ClassRecords {

    private String classID;
    private double prior;
    private int featureCount = 0;
    private List<SingleRecord> records = new ArrayList<>();
    private Features features;

    public ClassRecords(String d) {
        classID = d;
    }

    public String getClassID() {
        return this.classID;
    }

    public int getRecordsCount() {
        return records.size();
    }

    public void addOneRecord(SingleRecord sr) {
        records.add(sr);

    }

    public List<SingleRecord> getRecords() {
        return this.records;
    }

    public void setPrior(double prior) {
        this.prior = prior;
    }

    public double getPrior() {
        return prior;
    }

    public int getFeatureCount() {
        return featureCount;
    }

    public void constructFeatures() {
        features = new Features(this);
        features.constructFeatures();   
        //features.constructDistance();
    }
    
    public Map<Integer, Map<String, Double>> getFeatureCounter(){
        return features.getFeatureMap();
    }
    
//    public Map<Integer, Double> getFeatureDistance(){
//        return features.getFeatureDistance();
//    }
    
}
