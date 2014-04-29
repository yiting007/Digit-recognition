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
    private Map<Integer, Map<String, Double>> FeatureMap;   //(Feature index + value) --- Feature count
    private int INTEGER = 1;
    private int DOUBLE = 2;

    public Features(ClassRecords cr) {
        this.classRecord = cr;
        FeatureMap = new HashMap<>();
    }

    public void constructFeatures() {   //count the feature frenquency
        for (SingleRecord sr : classRecord.getRecords()) {
            List<String> features = sr.getFeatures();
            for (int i = 0; i < features.size(); i++) {
                Integer key = i;
                if (!FeatureMap.containsKey(i)) {
                    FeatureMap.put(key, new HashMap());
                }
                HashMap m = (HashMap) FeatureMap.get(key);
                String key2 = features.get(i);
                if (m.containsKey(key2)) {
                    m.put(key2, (double) m.get(key2) + 1.0);
                } else {
                    m.put(key2, 1.0);
                }
            }
        }
    }

    public Map<Integer, Map<String, Double>> getFeatureMap() {
        return FeatureMap;
    }
}
