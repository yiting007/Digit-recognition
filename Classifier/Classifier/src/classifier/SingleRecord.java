/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package classifier;

import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author Yiting
 */
public class SingleRecord {

    private ClassRecords classRecord;
    private List<String> feature = new ArrayList<>();

    public void setClassRecord(ClassRecords document) {
        this.classRecord = document;
    }

    public ClassRecords getClassRecord() {
        return classRecord;
    }

    public void addFeature(String p) {
        this.feature.add(p);
    }
    
    public int getRecordSize(){
        return feature.size();
    }

    public List<String> getFeatures() {
        return feature;
    }
}
