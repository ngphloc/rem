/**
 * SIM: MACHINE LEARNING ALGORITHMS FRAMEWORK
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: sim.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.rem.soco;

import java.io.Serializable;
import java.util.Set;

import net.hudup.core.Constants;

/**
 * This class represents parameter of Soco algorithm.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
@Deprecated
public class SocoParameter implements Cloneable, Serializable {

	
	/**
	 * Default serial version UID.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Row statistics of Soco algorithm. This is special case because statistics is considered as a part of parameter.
	 */
	protected SocoStatistics rowStat = new SocoStatistics();

	
	/**
	 * Column statistics of Soco algorithm. This is special case because statistics is considered as a part of parameter.
	 */
	protected SocoStatistics columnStat = new SocoStatistics();
	
	
	/**
	 * Default constructor.
	 */
	public SocoParameter() {
	
	}

	
	/**
	 * Default constructor with row statistics and column statistics.
	 * @param rowStat row statistics.
	 * @param columnStat column statistics.
	 */
	public SocoParameter(SocoStatistics rowStat, SocoStatistics columnStat) {
		this.rowStat = rowStat;
		this.columnStat = columnStat;
	}
	
	
	/**
	 * Getting row similarity at row id 1 and row id 2.
	 * @param rowId1 row id 1.
	 * @param rowId2 row id 2.
	 * @return row similarity at row id 1 and row id 2. Return {@link Constants#UNUSED} if row id 1 or row id 2 does not exist. 
	 */
	public double getRowSim(int rowId1, int rowId2) {
		return rowStat.getSim(rowId1, rowId2);
	}
	
	
	/**
	 * Getting row similarity at row id 1 and row id 2.
	 * @param rowId1 row id 1.
	 * @param rowId2 row id 2.
	 * @param sim specified row similarity.
	 */
	public void setRowSim(int rowId1, int rowId2, double sim) {
		rowStat.setSim(rowId1, rowId2, sim);
	}


	/**
	 * Checking whether the row similarity at row id 1 and row id 2 exists.
	 * @param rowId1 row id 1.
	 * @param rowId2 row id 2.
	 * @return whether the row similarity at row id 1 and row id 2 exists.
	 */
	public boolean containsRowSim(int rowId1, int rowId2) {
		return rowStat.containsSim(rowId1, rowId2);
	}
	
	
	/**
	 * Getting set of row id (s).
	 * @return set of row id (s).
	 */
	public Set<Integer> rowIds() {
		return rowStat.ids();
	}


	/**
	 * Getting column similarity at column id 1 and column id 2.
	 * @param columnId1 column id 1.
	 * @param columnId2 column id 2.
	 * @return column similarity at column id 1 and column id 2. Return {@link Constants#UNUSED} if column id 1 or column id 2 does not exist. 
	 */
	public double getColumnSim(int columnId1, int columnId2) {
		return columnStat.getSim(columnId1, columnId2);
	}
	
	
	/**
	 * Getting column similarity at column id 1 and column id 2.
	 * @param columnId1 column id 1.
	 * @param columnId2 column id 2.
	 * @param sim specified column similarity.
	 */
	public void setColumnSim(int columnId1, int columnId2, double sim) {
		columnStat.setSim(columnId1, columnId2, sim);
	}


	/**
	 * Checking whether the column similarity at column id 1 and column id 2 exists.
	 * @param columnId1 column id 1.
	 * @param columnId2 column id 2.
	 * @return whether the column similarity at column id 1 and column id 2 exists.
	 */
	public boolean containsColumnSim(int columnId1, int columnId2) {
		return columnStat.containsSim(columnId1, columnId2);
	}


	/**
	 * Getting set of column id (s).
	 * @return set of column id (s).
	 */
	public Set<Integer> columnIds() {
		return columnStat.ids();
	}


	/**
	 * Clearing this parameter.
	 */
	public void clear() {
		rowStat.clear();
		columnStat.clear();
	}


	@Override
	protected Object clone() {
		// TODO Auto-generated method stub
		SocoParameter newParameter = new SocoParameter();
		newParameter.rowStat = (SocoStatistics) this.rowStat.clone();
		newParameter.columnStat = (SocoStatistics) this.columnStat.clone();
		
		return newParameter;
	}


}
