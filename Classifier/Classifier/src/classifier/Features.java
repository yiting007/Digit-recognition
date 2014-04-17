/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package classifier;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 *
 * @author Yiting
 */
public class Features {

    private ClassRecords classRecord;
    private Map<String, Integer> FeatureMap;   //(Feature index + value) --- Feature count
    private Map<Integer, Double> FeatureDistance;    //Feature index --- Average value
    private int INTEGER = 1;
    private int DOUBLE = 2;

    public Features(ClassRecords cr) {
        this.classRecord = cr;
        FeatureMap = new HashMap<>();
        FeatureDistance = new HashMap<>();
    }

    public void constructFeatures() {   //count the feature frenquency
        for (SingleRecord sr : classRecord.getRecords()) {
            List<String> features = sr.getFeatures();
            for (int i = 0; i < features.size(); i++) {
                String key = String.valueOf(i) + "," + features.get(i);
                updateFeatureMap(FeatureMap, key, 1, INTEGER);
            }
        }
    }

    public void constructDistance() {
        Map<Integer, Double> valueList = new HashMap<>();
        for (SingleRecord sr : classRecord.getRecords()) {
            List<String> features = sr.getFeatures();
            for (int i = 0; i < features.size(); i++) {
                updateFeatureMap(valueList, i, Double.parseDouble(features.get(i)), DOUBLE);
            }
        }
        double num = classRecord.getRecords().get(0).getFeatures().size() * 1.0;
        for (Map.Entry<Integer, Double> entry : valueList.entrySet()) {
            Double value = entry.getValue();
            FeatureDistance.put(entry.getKey(), value/num);
        }
    }

    private void updateFeatureMap(Map m, Object key, Object value, int type) {
        if (!m.containsKey(key)) {
            m.put(key, value);
        } else {
            if (type == DOUBLE) {
                m.put(key, (Double) m.get(key) + (Double) value);
            }else{
                m.put(key, (Integer) m.get(key) + (Integer) value);
            }
        }
    }

    public Map<String, Integer> getFeatureMap() {
        return FeatureMap;
    }

    public Map<Integer, Double> getFeatureDistance() {
        return FeatureDistance;
    }
}
