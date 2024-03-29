/**
 * SIM: MACHINE LEARNING ALGORITHMS FRAMEWORK
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: sim.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.rem.regression;

import java.io.IOException;
import java.io.Serializable;
import java.io.Writer;
import java.util.List;

import net.hudup.core.Cloneable;
import net.hudup.core.Util;
import net.hudup.core.logistic.DSUtil;
import net.hudup.core.logistic.MathUtil;

/**
 * This class represents a data sample also a statistics for learning regression model.
 * 
 * @author Loc Nguyen
 * @version 1.0
 */
public class LargeStatistics implements Cloneable, Serializable {
	
	
	/**
	 * Default serial version UID.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Variable contains complete data of X.
	 */
	protected List<double[]> xData = Util.newList(); //1, x1, x2,..., x(n-1)
	
	
	/**
	 * Variable contains complete data of Z.
	 */
	protected List<double[]> zData = Util.newList(); //1, z
	
	
	/**
	 * Empty constructor.
	 */
	public LargeStatistics() {
		
	}
	
	
	/**
	 * Constructor with specified regressor data and response data. 
	 * @param xData specified regressor data: 1, x1, x2,..., x(n-1).
	 * @param zData specified response data: 1, z.
	 */
	public LargeStatistics(List<double[]> xData, List<double[]> zData) {
		this.xData = xData;
		this.zData = zData;
	}
	
	
	/**
	 * Getting data of X variables (X statistic): 1, x1, x2,..., x(n-1).
	 * @return data of X variables (X statistic): 1, x1, x2,..., x(n-1).
	 */
	public List<double[]> getXData() {
		return xData;
	}
	
	
	/**
	 * Getting data of Z variables (X statistic): 1, z.
	 * @return data of Z variables (X statistic): 1, z.
	 */
	public List<double[]> getZData() {
		return zData;
	}

	
	/**
	 * Getting X statistic as row vector: 1, x1, x2,..., x(n-1).
	 * @param row specified row.
	 * @return X statistic as row vector: 1, x1, x2,..., x(n-1).
	 */
	public double[] getXRowStatistic(int row) {
		return xData.get(row);
	}
	
	
	/**
	 * Getting X statistic as column vector.
	 * @param column specified column.
	 * @return X statistic as column vector.
	 */
	public List<Double> getXColumnStatistic(int column) {
		if (isEmpty())
			return Util.newList();
		
		List<Double> xColumnVector = Util.newList(xData.size());
		for (int i = 0; i < xData.size(); i++)
			xColumnVector.add(xData.get(i)[column]);
		
		return xColumnVector;
	}

	
	/**
	 * Getting Z statistic: z.
	 * @return Z statistic: z.
	 */
	public List<Double> getZStatistic() {
		if (isEmpty())
			return Util.newList();
		
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
	 * @param xData specified X data: 1, x1, x2,..., x(n-1).
	 * @param zData specified Z data: 1, z.
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
	 * Getting the size of this large statistics.
	 * @return the size of this large statistics.
	 */
	public int size() {
		return zData.size();
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


	@Override
	public Object clone() {
		List<double[]> xData = DSUtil.cloneDoubleArrayList(this.xData);
		List<double[]> zData = DSUtil.cloneDoubleArrayList(this.zData);
		
		return new LargeStatistics(xData, zData);
	}
	
	
	/**
	 * Saving this large statistics to specified writer.
	 * @param writer specified writer.
	 * @param decimal specified decimal.
	 * @return true if saving is successful.
	 * @throws IOException if any error raises
	 */
	public boolean save(Writer writer, int decimal) throws IOException {
		if (isEmpty())
			return false;
		
		StringBuffer columns = new StringBuffer();
		int n = xData.get(0).length;
		for (int i = 1; i < n; i++) {
			if (i > 1)
				columns.append(", ");
			columns.append("x" + i + "~real");
		}
		columns.append(", z~real");
		writer.write(columns.toString());

		int N = size();
		for (int i = 0; i < N; i++) {
			double[] xVector = xData.get(i);
			double[] zVector = zData.get(i);
			StringBuffer row = new StringBuffer(xVector.length + 1);
			
			row.append("\n");
			for (int j = 1; j < xVector.length; j++) {
				if (decimal > 0)
					row.append(MathUtil.format(xVector[j], decimal));
				else
					row.append(xVector[j]);

				row.append(", ");
			}
			if (decimal > 0)
				row.append(MathUtil.format(zVector[1], decimal));
			else
				row.append(zVector[1]);
			
			writer.write(row.toString());
		}

		return true;
	}
	
	
}
