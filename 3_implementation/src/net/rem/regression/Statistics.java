/**
 * SIM: MACHINE LEARNING ALGORITHMS FRAMEWORK
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: sim.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.rem.regression;

import java.io.Serializable;

import net.hudup.core.Constants;
import net.hudup.core.Util;

/**
 * This class represents a compound statistic.
 * @author Loc Nguyen
 * @version 1.0
 * 
 */
public class Statistics implements Serializable {

	
	/**
	 * Default serial version UID.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Statistic for Z variable: z.
	 */
	protected double zStatistic = Constants.UNUSED;
	
	
	/**
	 * Statistic for X variables: 1, x1, x2,..., x(n-1)
	 */
	protected double[] xStatistic = null;
	
	
	/**
	 * Constructor with specified statistic for Z variable and statistic for X variables.
	 * @param zStatistic statistic for Z variable: 1, x1, x2,..., x(n-1).
	 * @param xStatistic statistic for X variables: z. It must be not null but can be zero-length.
	 */
	public Statistics(double zStatistic, double[] xStatistic) {
		this.zStatistic = zStatistic;
		this.xStatistic = xStatistic;
	}
	
	
	/**
	 * Getting statistic for Z variable.
	 * @return statistic for Z variable.
	 */
	public double getZStatistic() {
		return zStatistic;
	}
	
	
	/**
	 * Getting statistic for X variables.
	 * @return statistic for X variables.
	 */
	public double[] getXStatistic() {
		return xStatistic;
	}
	
	
	/**
	 * Calculating mean of this statistics and other statistics.
	 * @param other other statistics.
	 * @return mean of this statistics and other statistics.
	 */
	public Statistics mean(Statistics other) {
		double zStatistic = (this.zStatistic + other.zStatistic) / 2.0;
		double[] xStatistic = new double[this.xStatistic.length];
		for (int i = 0; i < xStatistic.length; i++)
			xStatistic[i] = (this.xStatistic[i] + other.xStatistic[i]) / 2.0;
		
		return new Statistics(zStatistic, xStatistic);
	}
	
	
	/**
	 * Checking whether this statistics is valid.
	 * @return true if this statistics is valid.
	 */
	public boolean checkValid() {
		if (!Util.isUsed(zStatistic))
			return false;
		
		if (xStatistic == null || xStatistic.length == 0)
			return false;
		for (double x : xStatistic) {
			if (!Util.isUsed(x))
				return false;
		}
		
		return true;
	}
	
	
}
