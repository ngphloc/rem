/**
 * SIM: MACHINE LEARNING ALGORITHMS FRAMEWORK
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: sim.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.rem.soco;

import java.io.Serializable;
import java.util.Map;
import java.util.Set;

import net.hudup.core.Cloneable;
import net.hudup.core.Constants;
import net.hudup.core.Util;

/**
 * This class represents statistics of Soco algorithm.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
@Deprecated
public class SocoStatistics implements Cloneable, Serializable {

	
	/**
	 * Default serial version UID.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Map of similarities.
	 */
	protected Map<Integer, Map<Integer, Double>> simap = Util.newMap();
	
	
	/**
	 * Default constructor.
	 */
	public SocoStatistics() {
		// TODO Auto-generated constructor stub
	}
	

	/**
	 * Getting similarity at id 1 and id 2.
	 * @param id1 id 1.
	 * @param id2 id 2.
	 * @return similarity at id 1 and id 2. Return {@link Constants#UNUSED} if id 1 or id 2 does not exist. 
	 */
	public double getSim(int id1, int id2) {
		if (!simap.containsKey(id1)) return Constants.UNUSED;
		
		Map<Integer, Double> map = simap.get(id1);
		if (map.containsKey(id2))
			return map.get(id2);
		else
			return Constants.UNUSED;
	}
	
	
	/**
	 * Getting similarity at id 1 and id 2.
	 * @param id1 id 1.
	 * @param id2 id 2.
	 * @param sim specified similarity.
	 */
	public void setSim(int id1, int id2, double sim) {
		Map<Integer, Double> map = null;
		if (simap.containsKey(id1))
			map = simap.get(id1);
		else {
			map = Util.newMap();
			simap.put(id1, map);
		}
		map.put(id2, sim);
		
		if (id1 != id2) {
			Map<Integer, Double> inverseMap = null;
			if (simap.containsKey(id2))
				inverseMap = simap.get(id2);
			else {
				inverseMap = Util.newMap();
				simap.put(id2, inverseMap);
			}
			inverseMap.put(id1, sim);
		}
	}


	/**
	 * Checking whether the similarity at id 1 and id 2 exists.
	 * @param id1 id 1.
	 * @param id2 id 2.
	 * @return whether the similarity at id 1 and id 2 exists.
	 */
	public boolean containsSim(int id1, int id2) {
		if (!simap.containsKey(id1))
			return false;
		else
			return simap.get(id1).containsKey(id2);
	}
	
	
	/**
	 * Getting set of id (s).
	 * @return set of id (s).
	 */
	public Set<Integer> ids() {
		return simap.keySet();
	}
	
	
	/**
	 * Getting size of statistics.
	 * @return size of statistics.
	 */
	public int size() {
		return simap.size();
	}
	
	
	/**
	 * Clearing this statistics.
	 */
	public void clear() {
		simap.clear();
	}
	
	
	@Override
	public Object clone() {
		SocoStatistics newStat = new SocoStatistics();
		Set<Integer> ids1 = this.simap.keySet();
		for (int id1 : ids1) {
			Map<Integer, Double> map = this.simap.get(id1);
			Set<Integer> ids2 = map.keySet();
			Map<Integer, Double> newMap = Util.newMap();
			newStat.simap.put(id1, newMap);
			
			for (int id2 : ids2) {
				newMap.put(id2, map.get(id2));
			}
		}
		
		return newStat;
	}
	
	
	/**
	 * Testing whether terminated condition is satisfied with this statistics (as estimated statistics) and current statistics (specified statistics).
	 * @param threshold specified threshold.
	 * @param currentStat current statistics (specified statistics).
	 * @return whether terminated condition is satisfied with this statistics (as estimated statistics) and current statistics (specified statistics).
	 */
	protected boolean terminatedCondition(double threshold, SocoStatistics currentStat) {
		Set<Integer> ids1 = ids();
		Set<Integer> ids2 = Util.newSet(ids1.size());
		ids2.addAll(ids1);
		
		for (int id1 : ids1) {
			for (int id2 : ids2) {
				double estimatedSim = getSim(id1, id2);
				if (!Util.isUsed(estimatedSim)) continue;
				double currentSim = currentStat.getSim(id1, id2);
				if (!Util.isUsed(currentSim)) continue;
				
				if (notSatisfy(estimatedSim, currentSim, threshold))
					return false;
			}
		}
		
		return true;
	}
	
	
	/**
	 * Testing whether the deviation between estimated similarity and current similarity is not satisfied a threshold.
	 * @param estimatedSim estimated similarity.
	 * @param currentSim current similarity.
	 * @param threshold specified threshold.
	 * @return true if the deviation between estimated similarity and current similarity is not satisfied a threshold.
	 */
	protected boolean notSatisfy(double estimatedSim, double currentSim, double threshold) {
		return Math.abs(estimatedSim - currentSim) > threshold * Math.abs(currentSim);
	}

	
	/**
	 * Assign from other statistics;
	 * @param other other statistics;
	 */
	protected void assignFrom(SocoStatistics other) {
		this.simap = other.simap;
	}
	
	
}
