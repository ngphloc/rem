/**
 * SIM: MACHINE LEARNING ALGORITHMS FRAMEWORK
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: sim.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.rem.soco;

import java.rmi.RemoteException;
import java.util.List;
import java.util.Map;
import java.util.Set;

import net.hudup.core.Constants;
import net.hudup.core.Util;
import net.hudup.core.data.DataConfig;
import net.hudup.core.data.Dataset;
import net.hudup.core.data.Fetcher;
import net.hudup.core.data.Profile;
import net.hudup.core.data.Rating;
import net.hudup.core.data.RatingVector;
import net.hudup.core.logistic.BaseClass;
import net.hudup.core.logistic.DSUtil;
import net.rem.em.ExponentialEM;

/**
 * This class implements soft cosine similarity with missing values (Soco measure), based on expectation-maximization (EM) algorithm.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
@Deprecated
@BaseClass
public class Soco extends ExponentialEM {

	
	/**
	 * Default serial version UID.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Main unit.
	 */
	public final static String MAIN_UNIT_DEEFAULT = DataConfig.RATING_UNIT;
	
	
	/**
	 * Field of user rating matrix.
	 */
	public final static String USER_RATING_MATRIX_FIELD = "user_rating_matrix";

	
	/**
	 * Default user rating matrix.
	 */
	public final static boolean USER_RATING_MATRIX_DEEFAULT = true;

	
	/**
	 * Internal matrix.
	 */
	protected Map<Integer, RatingVector> matrix = Util.newMap();
	
	
	/**
	 * Internal transposed matrix.
	 */
	protected  Map<Integer, RatingVector> transposedMatrix = Util.newMap();

	
	/**
	 * Default constructor.
	 */
	public Soco() {

	}

	
	@Override
	protected Object fetchSample(Dataset dataset) {
		return dataset != null ? dataset.fetchSample() : null;
	}

	
	@Override
	public Object learnStart(Object... info) throws RemoteException {
		String mainUnit = config.getAsString(DataConfig.MAIN_UNIT);
		boolean prepared = false;
		if (mainUnit.equals(DataConfig.RATING_UNIT))
			prepared = prepareInternalDataByByRatingMatrix();
		else if (mainUnit.equals(DataConfig.SAMPLE_UNIT))
			prepared = prepareInternalDataBySample();
		
		if (!prepared) {
			clearInternalData();
			return null;
		}

		return super.learnStart(info);
	}


	/**
	 * Preparing data by rating matrix.
	 * @return true if data preparation is successful.
	 * @throws RemoteException if any error raises.
	 */
	protected boolean prepareInternalDataByByRatingMatrix() throws RemoteException {
		clearInternalData();
		this.matrix = Util.newMap();
		this.transposedMatrix = Util.newMap();
		
		boolean isUser = config.getAsBoolean(USER_RATING_MATRIX_FIELD);
		Fetcher<RatingVector> vRatings = isUser ? dataset.fetchUserRatings() : dataset.fetchItemRatings();
		while (vRatings.next()) {
			RatingVector vRating = vRatings.pick();
			if (vRating == null) continue;
			
			Set<Integer> columnIds = vRating.fieldIds(true);
			if (columnIds.size() == 0) continue;
			RatingVector vNewRating = new RatingVector(vRating.id());
			this.matrix.put(vNewRating.id(), vNewRating);
			for (int columnId : columnIds) {
				Rating rating = vRating.get(columnId);
				
				vNewRating.put(columnId, rating);
				
				RatingVector vNewTransposedRating = null;
				if (this.transposedMatrix.containsKey(columnId))
					vNewTransposedRating = this.transposedMatrix.get(columnId);
				else {
					vNewTransposedRating = new RatingVector(columnId);
					this.transposedMatrix.put(columnId, vNewTransposedRating);
				}
				vNewTransposedRating.put(vNewRating.id(), rating);
			}
		}
		vRatings.close();
		
		return this.matrix.size() > 0 && this.transposedMatrix.size() > 0;
	}

	
	/**
	 * Preparing data by sample.
	 * @return true if data preparation is successful.
	 * @throws RemoteException if any error raises.
	 */
	@SuppressWarnings("unchecked")
	protected boolean prepareInternalDataBySample() throws RemoteException {
		clearInternalData();
		
		this.matrix = Util.newMap();
		this.transposedMatrix = Util.newMap();
		int rowId = 1;
		while (((Fetcher<Profile>)sample).next()) {
			Profile profile = ((Fetcher<Profile>)sample).pick();
			if (profile == null || profile.isAllMissing())
				continue;
			
			RatingVector vRating = new RatingVector(rowId);
			this.matrix.put(vRating.id(), vRating);
			for (int i = 0; i < profile.getAttCount(); i++) {
				double value = profile.getValueAsReal(i);
				if (!Util.isUsed(value)) continue;
				Rating rating = new Rating(value);
				
				int columnId = i;
				vRating.put(columnId, rating);
				
				RatingVector vTransposedRating = null;
				if (this.transposedMatrix.containsKey(columnId))
					vTransposedRating = this.transposedMatrix.get(columnId);
				else {
					vTransposedRating = new RatingVector(columnId);
					this.transposedMatrix.put(columnId, vTransposedRating);
				}
				vTransposedRating.put(rowId, rating);
			}
			
			rowId++;
		}
		((Fetcher<Profile>)sample).reset();
		
		return this.matrix.size() > 0 && this.transposedMatrix.size() > 0;
	}
	
	
	/**
	 * Clear all internal data.
	 */
	protected void clearInternalData() {
		this.currentIteration = 0;
		
		if (this.currentParameter != null && (this.currentParameter instanceof SocoParameter))
			((SocoParameter)this.currentParameter).clear();
		if (this.estimatedParameter != null && this.estimatedParameter != this.currentParameter && (this.currentParameter instanceof SocoParameter))
			((SocoParameter)this.estimatedParameter).clear();
		this.currentParameter = this.estimatedParameter = null;
		
		if (this.statistics != null && (this.statistics instanceof SocoStatistics))
			((SocoStatistics)this.statistics).clear();
		this.statistics = null;

		if (this.matrix != null)
			this.matrix.clear();
		this.matrix = null;
		
		if (this.transposedMatrix != null)
			this.transposedMatrix.clear();
		this.transposedMatrix = null;
	}

	
	/**
	 * Calculating cosine similarities from rating vector matrix.
	 * @param matrix rating vector matrix.
	 * @param outStat output statistics that contains cosine similarities from rating vector matrix.
	 */
	private void sim(Map<Integer, RatingVector> matrix, SocoStatistics outStat) {
		List<Integer> ids = Util.newList();
		ids.addAll(matrix.keySet());
		for (int i = 0; i < ids.size(); i++) {
			int id1 = ids.get(i);
			RatingVector v1 = matrix.get(id1);
			
			for (int j = 0; j < ids.size(); j++) {
				int id2 = ids.get(j);
				RatingVector v2 = matrix.get(id2);
				double sim = sim(v1, v2);
				
				sim = Util.isUsed(sim)? sim : 0;
				outStat.setSim(id1, id2, sim);
			}
		}
	}
	
	
	/**
	 * Calculating similarity (often cosine) of two rating vectors.
	 * @param vRating1 rating vector 1.
	 * @param vRating2 rating vector 2.
	 * @return similarity (often cosine) of two rating vectors.
	 */
	protected double sim(RatingVector vRating1, RatingVector vRating2) {
		return vRating1.cosine(vRating2);
	}
	
	
	/**
	 * Calculating soft cosine given rating vector matrix and dual statistics.
	 * @param matrix given rating vector matrix.
	 * @param columnIds column identifiers.
	 * @param dualStat dual statistics.
	 * @return statistics as soft cosine given rating vector matrix and dual statistics.
	 */
	private static SocoStatistics softCosine(Map<Integer, RatingVector> matrix, Set<Integer> columnIds, SocoStatistics dualStat) {
		if (matrix == null || matrix.size() == 0 || dualStat == null || dualStat.size() == 0 || columnIds.size() == 0)
			return null;
		List<Integer> rowIds = Util.newList();
		rowIds.addAll(matrix.keySet());
		
		SocoStatistics stat = new SocoStatistics();
		Set<Integer> columnIds1 = columnIds;
		Set<Integer> columnIds2 = Util.newSet(columnIds.size());
		columnIds2.addAll(columnIds);
		for (int i = 0; i < rowIds.size(); i++) {
			int rowId1 = rowIds.get(i);
			RatingVector vRating1 = matrix.get(rowId1);
			
			for (int j = i; j < rowIds.size(); j++) {
				int rowId2 = rowIds.get(j);
				RatingVector vRating2 = matrix.get(rowId2);
				
				double product = 0;
				double length1 = 0;
				double length2 = 0;
				for (int columnId1 : columnIds1) {
					double value11 = vRating1.isRated(columnId1) ? vRating1.get(columnId1).value : Constants.UNUSED;
					double value21 = vRating2.isRated(columnId1) ? vRating2.get(columnId1).value : Constants.UNUSED;
					for (int columnId2 : columnIds2) {
						double value12 = vRating1.isRated(columnId2) ? vRating1.get(columnId2).value : Constants.UNUSED;
						double value22 = vRating2.isRated(columnId2) ? vRating2.get(columnId2).value : Constants.UNUSED;
						
						double sim = dualStat.getSim(columnId1, columnId2);
						if (Util.isUsed(value11) && Util.isUsed(value22))
							product += sim*value11*value22;
						
						if (Util.isUsed(value11) && Util.isUsed(value12))
							length1 += sim*value11*value12;
						
						if (Util.isUsed(value21) && Util.isUsed(value22))
							length2 += sim*value21*value22;
					}
				}
				
				double sim = product / Math.sqrt(length1*length2);				
				sim = Util.isUsed(sim)? sim : 0;
				stat.setSim(rowId1, rowId2, sim);
			}
			
		}
		
		return stat;
	}
	
	
	@Override
	protected Object expectation(Object currentParameter, Object... info) throws RemoteException {
		SocoStatistics rowStat = ((SocoParameter)currentParameter).rowStat;
		SocoStatistics columnStat = softCosine(this.transposedMatrix, rowIds(), rowStat);
		return columnStat;
	}

	
	@Override
	protected Object maximization(Object currentStatistic, Object... info) throws RemoteException {
		SocoStatistics rowStat = softCosine(this.matrix, columnIds(), (SocoStatistics)currentStatistic);
		if (rowStat == null)
			return null;
		else
			return new SocoParameter(rowStat, (SocoStatistics)currentStatistic);
	}

	
	@Override
	protected Object initializeParameter() {
		this.statistics = new SocoStatistics();
		sim(this.transposedMatrix, (SocoStatistics)this.statistics);

		SocoStatistics rowStat = softCosine(this.matrix, columnIds(), (SocoStatistics)this.statistics);
		if (rowStat == null)
			return null;
		else
			return new SocoParameter(rowStat, (SocoStatistics)this.statistics);
	}

	
	@Override
	protected boolean terminatedCondition(Object estimatedParameter, Object currentParameter, Object previousParameter,
			Object... info) {
		double threshold = getConfig().getAsReal(EM_EPSILON_FIELD);
		boolean terminated = (((SocoParameter)estimatedParameter).rowStat).terminatedCondition(
				threshold, ((SocoParameter)currentParameter).rowStat);
		terminated = terminated && (((SocoParameter)estimatedParameter).columnStat).terminatedCondition(
				threshold, ((SocoParameter)currentParameter).columnStat);
				
		return terminated;
	}

	
	/**
	 * Getting row identifiers.
	 * @return row identifiers.
	 */
	public Set<Integer> rowIds() {
		if (this.matrix == null)
			return Util.newSet();
		else
			return this.matrix.keySet();
	}
	
	
	/**
	 * Getting column identifiers.
	 * @return column identifiers.
	 */
	public Set<Integer> columnIds() {
		if (this.transposedMatrix == null)
			return Util.newSet();
		else
			return this.transposedMatrix.keySet();
	}

	
	/**
	 * Getting similarity at row id 1 and row id 2.
	 * @param rowId1 row id 1.
	 * @param rowId2 row id 2.
	 * @return similarity at row id 1 and row id 2. Return {@link Constants#UNUSED} if row id 1 or row id 2 does not exist.
	 * @throws RemoteException if any error raises. 
	 */
	public synchronized double getRowSim(int rowId1, int rowId2) throws RemoteException {
		SocoParameter parameter = (SocoParameter)getParameter();
		if (parameter == null)
			return Constants.UNUSED;
		else
			return parameter.getRowSim(rowId1, rowId2);
	}
	
	
	/**
	 * Getting similarity at column id 1 and column id 2.
	 * @param columnId1 column id 1.
	 * @param columnId2 column id 2.
	 * @return similarity at column id 1 and column id 2. Return {@link Constants#UNUSED} if column id 1 or column id 2 does not exist.
	 * @throws RemoteException if any error raises. 
	 */
	public synchronized double getColumnSim(int columnId1, int columnId2) throws RemoteException {
		SocoParameter parameter = (SocoParameter)getParameter();
		if (parameter == null)
			return Constants.UNUSED;
		else
			return parameter.getColumnSim(columnId1, columnId2);
	}

	
	@Override
	public String getName() {
		return "soco";
	}

	
	@Override
	public synchronized Object execute(Object input) throws RemoteException {
		List<Double> ids = DSUtil.toDoubleList(input, true);
		if (ids.size() < 2)
			return null;
		else
			return getRowSim((int)ids.get(0).doubleValue(), (int)ids.get(1).doubleValue());
	}

	
	@Override
	public String parameterToShownText(Object parameter, Object... info) throws RemoteException {
		return "Too large to show";
	}

	
	@Override
	public String getDescription() throws RemoteException {
		return "Soft cosine similarity with missing values, based on expectation-maximization (EM) algorithm";
	}


	@Override
	public DataConfig createDefaultConfig() {
		DataConfig config = super.createDefaultConfig();
		if (!config.containsKey(DataConfig.MAIN_UNIT))
			config.put(DataConfig.MAIN_UNIT, MAIN_UNIT_DEEFAULT);
		
		config.put(USER_RATING_MATRIX_FIELD, USER_RATING_MATRIX_DEEFAULT);
		
		return config;
	}

	
}
