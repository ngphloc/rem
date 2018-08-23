package net.hudup.regression.em;

import java.util.List;

import net.hudup.core.Util;

/**
 * This class represents a data sample also a statistics for learning regression model.
 * 
 * @author Loc Nguyen
 * @version 1.0
 */
public class LargeStatistics {
	
	
	/**
	 * Variable contains complete data of X.
	 */
	protected List<double[]> xData = Util.newList(); //1, x1, x2,..., x(n-1)
	
	
	/**
	 * Variable contains complete data of Z.
	 */
	protected List<double[]> zData = Util.newList(); //1, z
	
	
	/**
	 * Constructor with specified regressor data and response data. 
	 * @param xData specified regressor data
	 * @param zData specified response data.
	 */
	public LargeStatistics(List<double[]> xData, List<double[]> zData) {
		this.xData = xData;
		this.zData = zData;
	}
	
	
	/**
	 * Getting data of X variables (X statistic).
	 * @return data of X variables (X statistic).
	 */
	public List<double[]> getXData() {
		return xData;
	}
	
	
	/**
	 * Getting data of X variables (X statistic).
	 * @return data of X variables (X statistic).
	 */
	public List<double[]> getZData() {
		return zData;
	}

	
	/**
	 * Getting X statistic as row vector.
	 * @param row specified row.
	 * @return X statistic as row vector.
	 */
	public double[] getXRowStatistic(int row) {
		return xData.get(row);
	}
	
	
	/**
	 * Getting X statistic as column vector.
	 * @param column specified column.
	 * @return X statistic as column vector.
	 */
	public double[] getXColumnStatistic(int column) {
		if (isEmpty())
			return null;
		
		double[] xColumnVector = new double[xData.size()];
		for (int i = 0; i < xData.size(); i++)
			xColumnVector[i] = xData.get(i)[column];
		
		return xColumnVector;
	}

	
	/**
	 * Getting Z statistic.
	 * @return Z statistic.
	 */
	public List<Double> getZStatistic() {
		if (isEmpty())
			return null;
		
		List<Double> zVector = Util.newList(zData.size());
		for (int i = 0; i < zData.size(); i++)
			zVector.add(zData.get(i)[1]);
		
		return zVector;
	}

	
	/**
	 * Getting both X statistic and Z statistic.
	 * @param row specified row.
	 * @return {@link Statistics} containing both X statistic and Z statistic.
	 */
	public Statistics getStatistic(int row) {
		if (isEmpty())
			return null;
		
		double[] xStatistic = getXRowStatistic(row);
		List<Double> zStatistic = getZStatistic();
		if (xStatistic == null || zStatistic == null || xStatistic.length == 0 || zStatistic.size() == 0)
			return null;
		else
			return new Statistics(zStatistic.get(row), xStatistic);
	}
	
	
	/**
	 * Checking whether this statistics is valid.
	 * @return true if this statistics is valid.
	 */
	public boolean checkValid() {
		return checkValid(this.xData, this.zData);
	}
	
	
	/**
	 * Checking whether specified X data and Z data are valid.
	 * @param xData specified X data.
	 * @param zData specified Z data.
	 * @return true if both X data and Z data are valid.
	 */
	private static boolean checkValid(List<double[]> xData, List<double[]> zData) {
		if (xData == null || zData == null || xData.size() != zData.size())
			return false;
		else
			return true;
	}

	
	/**
	 * Checking whether this statistics is empty.
	 * @return true this statistics is empty.
	 */
	public boolean isEmpty() {
		if (xData == null || zData == null || xData.size() == 0 || zData.size() == 0)
			return true;
		else
			return false;
	}
	
	
	/**
	 * Clear data.
	 */
	public void clear() {
		if (xData != null) {
			xData.clear();
			xData = null;
		}
		if (zData != null) {
			zData.clear();
			zData = null;
		}
	}
	
	
}
