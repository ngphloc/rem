/**
 * SIM: MACHINE LEARNING ALGORITHMS FRAMEWORK
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: sim.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.rem.regression.em;

import static net.rem.regression.em.REMImpl.CALC_VARIANCE_FIELD;

import java.rmi.RemoteException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import net.hudup.core.Constants;
import net.hudup.core.Util;
import net.hudup.core.alg.DuplicatableAlg;
import net.hudup.core.alg.NoteAlg;
import net.hudup.core.data.AttributeList;
import net.hudup.core.data.DataConfig;
import net.hudup.core.data.Fetcher;
import net.hudup.core.data.Profile;
import net.hudup.core.logistic.DSUtil;
import net.hudup.core.logistic.LogUtil;
import net.hudup.core.logistic.MathUtil;
import net.rem.regression.LargeStatistics;

/**
 * This class implements the mixture regression model.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class DefaultMixtureREM extends AbstractMixtureREM implements DuplicatableAlg, NoteAlg {

	
	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Name of cluster number field.
	 */
	public final static String COMP_NUMBER_FIELD = "mixrem_comp_number";

	
	/**
	 * Default number of cluster.
	 */
	public final static int COMP_NUMBER_DEFAULT = 1;

	
	/**
	 * Name of previous parameters field.
	 */
	public final static String PREV_PARAMS_FIELD = "mixrem_prev_parameters";

	
	/**
	 * Vicinity for calculating probability from probability density function.
	 */
	public final static double VICINITY = 0.001;

	
	/**
	 * Internal data.
	 */
	protected LargeStatistics data = null;
	
	
	/**
	 * Indices for X data.
	 */
	protected List<Object[]> xIndices = Util.newList(); //Object list for parsing mathematical expressions in the most general case.
	
	
	/**
	 * Indices for Z data.
	 */
	protected List<Object[]> zIndices = Util.newList(); //Object list for parsing mathematical expressions in the most general case.
	
	
	/**
	 * Attribute list for all variables: all X, Y, and z.
	 * This variable is also used as the indicator of successful learning (not null).
	 */
	protected AttributeList attList = null;

	
	@Override
	protected boolean prepareInternalData(AbstractMixtureREM other) throws RemoteException {
		if (other instanceof DefaultMixtureREM) {
			DefaultMixtureREM mixREM = (DefaultMixtureREM)other;
			return prepareInternalData(mixREM.xIndices, mixREM.zIndices, mixREM.attList, mixREM.data);
		}
		else
			return super.prepareInternalData(other);
	}


	@Override
	protected boolean prepareInternalData(Fetcher<Profile> inputSample) throws RemoteException {
		clearInternalData();

		REMImpl tempEM = new REMImpl();
		tempEM.getConfig().put(RM_INDICES_FIELD, this.getConfig().get(RM_INDICES_FIELD));
		if (!tempEM.prepareInternalData(inputSample))
			return false;
		else {
			boolean result = prepareInternalData(tempEM.xIndices, tempEM.zIndices, tempEM.attList, tempEM.data);
			//Setting internal variables of temporal REM to be null so that it is possible to prevent unexpected finalize() of REMImpl in future.
			tempEM.xIndices = tempEM.zIndices = null; tempEM.attList = null; tempEM.data = null;
			return result;
		}
	}

	
	/**
	 * Setting internal data.
	 * @param xIndices specified X indices.
	 * @param zIndices specified Z indices.
	 * @param attList specified attribute list.
	 * @param data specified data.
	 * @return true if setting successful.
	 * @throws RemoteException if any error raises.
	 */
	private boolean prepareInternalData(List<Object[]> xIndices, List<Object[]> zIndices, AttributeList attList, LargeStatistics data) throws RemoteException {
		clearInternalData();
		
		this.xIndices = xIndices;
		this.zIndices = zIndices;
		this.attList = attList;
		this.data = data;

		int K = getConfig().getAsInt(COMP_NUMBER_FIELD);
		K = K <= 0 ? 1 : K;
		this.rems = Util.newList(K);
		for (int k = 0; k < K; k++) {
			REMImpl rem = createREM();
			rem.prepareInternalData(this.xIndices, this.zIndices, this.attList, this.data);
			this.rems.add(rem);
		}
		
		return true;
	}
	
	
	@Override
	protected void clearInternalData() throws RemoteException {
		super.clearInternalData();
		this.xIndices.clear();
		this.zIndices.clear();
		this.attList = null;
		if (this.data != null)
			this.data.clear();
		this.data = null;
	}


	@Override
	protected REMImpl createREM() {
		REMImpl rem = super.createREM();
		rem.getConfig().put(EM_EPSILON_FIELD, this.getConfig().get(EM_EPSILON_FIELD));
		rem.getConfig().put(EM_MAX_ITERATION_FIELD, this.getConfig().get(EM_MAX_ITERATION_FIELD));
		rem.getConfig().put(RM_INDICES_FIELD, this.getConfig().get(RM_INDICES_FIELD));
		rem.getConfig().put(REMImpl.ESTIMATE_MODE_FIELD, this.getConfig().get(REMImpl.ESTIMATE_MODE_FIELD));
		rem.getConfig().put(CALC_VARIANCE_FIELD, true);
		return rem;
	}


	@Override
	protected Object expectation(Object currentParameter, Object... info) throws RemoteException {
		@SuppressWarnings("unchecked")
		List<ExchangedParameter> parameters = (List<ExchangedParameter>)currentParameter;
		@SuppressWarnings("unchecked")
		List<LargeStatistics> stats = (List<LargeStatistics>)super.expectation(currentParameter, info);
		if (stats == null) return null;
		
		//Adjusting large statistics.
		int N = stats.get(0).getZData().size(); //Suppose all models have the same data.
		int n = stats.get(0).getXData().get(0).length;  //Suppose all models have the same data.
		List<double[]> xData = Util.newList(N);
		List<double[]> zData = Util.newList(N);
		for (int i = 0; i < N; i++) {
			double[] xVector = new double[n];
			Arrays.fill(xVector, 0.0);
			xVector[0] = 1;
			xData.add(xVector);
			
			double[] zVector = new double[2];
			zVector[0] = 1;
			zVector[1] = 0;
			zData.add(zVector);
		}
		
		
		String estimateMode = getConfig().getAsString(REMImpl.ESTIMATE_MODE_FIELD);
		int K = this.rems.size();
		if (estimateMode.equals(REMImpl.REVERSIBLE)) {
			for (int k = 0; k < K; k++) {
				double coeff = parameters.get(k).getCoeff();
				LargeStatistics stat = stats.get(k);
				
				for (int i = 0; i < N; i++) {
					double[] zVector = zData.get(i);
					double zValue = stat.getZData().get(i)[1];
					if (!Util.isUsed(this.data.getZData().get(i)[1]))
						zVector[1] += coeff * zValue;
					else
						zVector[1] = zValue; 
	
					double[] xVector = xData.get(i);
					for (int j = 1; j < n; j++) {
						double xValue = stat.getXData().get(i)[j];
						if (!Util.isUsed(this.data.getXData().get(i)[j]))
							xVector[j] += coeff * xValue; // This assignment is right with assumption of same P(Y=k).
						else
							xVector[j] = xValue;
					}
				}
			}
		} //End if (estimateMode.equals(REMImpl.REVERSIBLE))
		else if (estimateMode.equals(REMImpl.GAUSSIAN)) {
			List<List<Double>> weights = Util.newList(K); //K lists of weights.
			for (int k = 0; k < K; k++) {
				ExchangedParameter parameter = parameters.get(k); 
				LargeStatistics stat = stats.get(k);
				List<Double> kWeights = Util.newList(N); //The kth list of weights.
				weights.add(kWeights);
				
				double coeff = parameter.getCoeff();
				List<Double> mean = parameter.getXNormalDisParameter().getMean();
				List<double[]> variance = parameter.getXNormalDisParameter().getVariance();
				for (int i = 0; i < N; i++) {
					if (!Util.isUsedAll(this.data.getXData().get(i))) {
						double[] xVector = stat.getXData().get(i);
						double pdf = ExchangedParameter.normalPDF(
							DSUtil.toDoubleList(Arrays.copyOfRange(xVector, 1, xVector.length)),
							mean,
							variance);
						kWeights.add(coeff*pdf);
					}
					else
						kWeights.add(1.0); //Not necessary to calculate the probabilities.
				}
			}
			
			List<List<Double>> newCoeffs = Util.newList(N);
			for (int i = 0; i < N; i++) {
				List<Double> kNewCoeffs = Util.newList(K);
				newCoeffs.add(kNewCoeffs);
				
				double kSumCoeffs = 0;
				for (int k = 0; k < K; k++) {
					kSumCoeffs += weights.get(k).get(i);
				}
				
				if (kSumCoeffs != 0 && Util.isUsed(kSumCoeffs)) {
					for (int k = 0; k < K; k++) {
						kNewCoeffs.add(weights.get(k).get(i) / kSumCoeffs);
					}
				}
				else {
					double w = 1.0 / (double)K;
					for (int k = 0; k < K; k++) {
						kNewCoeffs.add(w);
					}
				}
			}
			weights.clear();

			for (int k = 0; k < K; k++) {
				LargeStatistics stat = stats.get(k);
				for (int i = 0; i < N; i++) {
					double[] zVector = zData.get(i);
					double zValue = stat.getZData().get(i)[1];
					if (!Util.isUsed(this.data.getZData().get(i)[1]))
						zVector[1] += newCoeffs.get(i).get(k) * zValue;
					else
						zVector[1] = zValue; 
					
					double[] xVector = xData.get(i);
					for (int j = 1; j < n; j++) {
						double xValue = stat.getXData().get(i)[j];
						if (!Util.isUsed(this.data.getXData().get(i)[j]))
							xVector[j] += newCoeffs.get(i).get(k) * xValue; // This assignment is right with assumption of same P(Y=k).
						else
							xVector[j] = xValue;
					}
				}
			}
		} //End if (estimateMode.equals(REMImpl.GAUSSIAN))
		
		
		//All regression models have the same large statistics.
		stats.clear();
		LargeStatistics stat = new LargeStatistics(xData, zData);
		for (REMImpl rem : this.rems) {
			rem.setStatistics(stat);
			stats.add(stat);
		}
		
		return stats;
	}

	
	@Override
	protected Object maximization(Object currentStatistic, Object... info) throws RemoteException {
		if (currentStatistic == null) return null;
		
		@SuppressWarnings("unchecked")
		List<LargeStatistics> stats = (List<LargeStatistics>)currentStatistic;
		List<ExchangedParameter> parameters = Util.newList(this.rems.size());
		List<List<Double>> condProbs = Util.newList(this.rems.size()); //K lists of conditional probabilities.
		
		int K = this.rems.size();
		for (int k = 0; k < K; k++) {
			LargeStatistics stat = stats.get(k); //Each REM has particular large statistics. 
			int N = stat.getZStatistic().size();
			List<Double> kCondProbs = Util.newList(N); //The kth list of conditional probabilities.
			condProbs.add(kCondProbs);
			
			for (int i = 0; i < N; i++) {
				List<double[]> xData = Util.newList(K);
				List<double[]> zData = Util.newList(K);
				
				for (int j = 0; j < K; j++) {
					xData.add(stat.getXData().get(i));
					zData.add(stat.getZData().get(i));
				}
				
				@SuppressWarnings("unchecked")
				List<Double> probs = ExchangedParameter.normalZCondProbs((List<ExchangedParameter>)this.getCurrentParameter(),
						xData, zData);
				kCondProbs.add(probs.get(k));
			}
		}
		
		for (int k = 0; k < K; k++) {
			REMImpl rem = this.rems.get(k);
			LargeStatistics stat = stats.get(k);

			ExchangedParameter parameter = (ExchangedParameter)rem.maximization(stat, condProbs.get(k));
			rem.setEstimatedParameter(parameter);
			parameters.add(parameter);
		}
		
		return parameters;
	}


	@Override
	protected Object initializeParameter() {
		return initializeParameter0(config.getAsBoolean(INITIALIZE_GIVEBACK_FIELD));
	}
	
	
	/**
	 * Initialization method of this class changes internal data.
	 * This method improves the initialization process so that sub-models do not coincide when regression coefficients are made different.
	 * The diversity is important to converge best solutions.
	 * @param giveBack if true, the random record is given back to original sample.
	 * @return initialized parameter at the first iteration of EM process.
	 */
	@SuppressWarnings("unchecked")
	private Object initializeParameter0(boolean giveBack) {
		List<ExchangedParameter> prevParameters = Util.newList();
		if (getConfig().containsKey(PREV_PARAMS_FIELD))
			prevParameters = (List<ExchangedParameter>)getConfig().get(PREV_PARAMS_FIELD);
		
		List<ExchangedParameter> parameters = Util.newList(this.rems.size());
		LargeStatistics completeData = REMImpl.getCompleteData(this.data);
		int recordNumber = 0;
		if (completeData == null)
			recordNumber = 0;
		else if (giveBack)
			recordNumber = completeData.getZData().size();
		else if (this.rems.size() == prevParameters.size())
			recordNumber = 0;
		else if (completeData.getZData().size() >= this.rems.size() - prevParameters.size())
			recordNumber = completeData.getZData().size() / (this.rems.size() - prevParameters.size());
		else
			recordNumber = 0;
				
		for (int k = 0; k < this.rems.size(); k++) {
			REMImpl rem = this.rems.get(k);
			ExchangedParameter parameter = null;
			
			if (k < prevParameters.size()) {
				parameter = prevParameters.get(k);
			}
			else {
				if (recordNumber > 0) {
					try {
						LargeStatistics compSample = randomSampling(completeData, recordNumber, giveBack);
						parameter = (ExchangedParameter) rem.maximization(compSample);
						compSample.clear();
						if (parameter != null) {
							for (int j = 0; j < k; j++) {
								if (parameter.alphaEquals(this.rems.get(j).getExchangedParameter())) { //Avoid same alphas
									parameter = null;
									break;
								}
							}
						}
					}
					catch (Throwable e) {
						parameter = null;
						LogUtil.trace(e);
					}
				}
				else {
					parameter = null;
 				}
				
				if (parameter == null) {
					while (true) { // This loop avoids same alpha.
						parameter = rem.initializeParameterWithoutData(this.data.getXData().get(0).length - 1, true);
						boolean breakhere = true;
						for (int j = 0; j < k; j++) {
							if (parameter.alphaEquals(this.rems.get(j).getExchangedParameter())) {
								breakhere = false;
								break;
							}
						}
						if (breakhere) break;
					}
					parameter.setZVariance(1.0);
				}
			}
			
			parameter.setCoeff(1.0 / (double)this.rems.size());
			
			rem.setEstimatedParameter(parameter);
			rem.setCurrentParameter(parameter);
			rem.setPreviousParameter(null);
			rem.setStatistics(null);
			rem.setCurrentIteration(this.getCurrentIteration());

			parameters.add(parameter);
		}
		
		return parameters;
	}


	@Override
	public synchronized Object execute(Object input) throws RemoteException {
		double[] xStatistic = extractRegressorValues(input); //because all sub-model has the same attribute list.
		return executeByXStatistic(xStatistic);
	}


	/**
	 * Getting the fitness criterion of this model given large statistics.
	 * @param stat given large statistics.
	 * @return the fitness criterion of this model given large statistics. Return NaN if any error raises.
	 * @throws RemoteException if any error raises.
	 */
	public synchronized double getFitness(LargeStatistics stat) throws RemoteException {
		@SuppressWarnings("unchecked")
		List<ExchangedParameter> parameters = (List<ExchangedParameter>)getParameter();
		if (stat == null || parameters == null || parameters.size() == 0)
			return Constants.UNUSED;
		
		int N = stat.getZData().size();
		if (N == 0) return Constants.UNUSED;
		double fitness = 0.0;
		for (int i = 0; i < N; i++) {
			double[] xVector = stat.getXData().get(i);
			double[] zVector = stat.getZData().get(i);
			
			List<Double> pdfValues = ExchangedParameter.normalZPDF(parameters, xVector, zVector);
			double[] max = MathUtil.findExtremeValue(pdfValues, true);
			if (max != null)
				fitness += max[0];
		}
		
		return fitness / (double)N;
	}
	
	
	/**
	 * Getting the fitness criterion of this model.
	 * @return the fitness criterion of this model.
	 * @throws RemoteException if any error raises.
	 */
	public synchronized double getFitness() throws RemoteException {
		if (this.rems == null || this.rems.size() == 0)
			return 0;
		else
			return getFitness(this.getLargeStatistics()); // Because all REMs have the same large statistics.
	}

	
	/**
	 * Extracting clusters.
	 * @return extracted clusters.
	 * @throws RemoteException if any error raises.
	 */
	public synchronized List<LargeStatistics> extractClusters() throws RemoteException {
		if (this.rems == null || this.rems.size() == 0 || this.data == null || this.data.size() == 0)
			return Util.newList();
		@SuppressWarnings("unchecked")
		List<ExchangedParameter> parameters = (List<ExchangedParameter>)getParameter();
		if (parameters == null || parameters.size() == 0)
			return Util.newList();
		
		int K = this.rems.size();
		List<LargeStatistics> clusters = Util.newList(K);
		for (int k = 0; k < K; k++)
			clusters.add(new LargeStatistics());

		// Because all REMs have the same large statistics.
		LargeStatistics stat = this.getLargeStatistics();
		int N = stat.size();
		for (int i = 0; i < N; i++) {
			double[] xVector = stat.getXData().get(i);
			double[] zVector = stat.getZData().get(i);
			
			int maxK = getComponent(xVector, zVector);
			if (maxK >= 0) {
				clusters.get(maxK).getXData().add(xVector);
				clusters.get(maxK).getZData().add(zVector);
			}
		}
		
		return clusters;
	}
	
	
	/**
	 * Retrieving component of given regressors and response value.
	 * @param xVector regressor values. 
	 * @param zVector response value.
	 * @return component of given regressors and response value.
	 * @throws RemoteException if any error raises.
	 */
	protected synchronized int getComponent(double[] xVector, double[] zVector) throws RemoteException {
		@SuppressWarnings("unchecked")
		List<ExchangedParameter> parameters = (List<ExchangedParameter>)getParameter();
		if (parameters == null || parameters.size() == 0)
			return -1;
		
		int K = this.rems.size();
		List<double[]> xData = Util.newList(K);
		List<double[]> zData = Util.newList(K);
		for (int k = 0; k < K; k++) {
			xData.add(xVector);
			zData.add(zVector);
		}
		
		List<Double> condProbs = ExchangedParameter.normalZCondProbs(parameters, xData, zData);
		double maxProb = -1;
		int maxK = -1;
		for (int k = 0; k < K; k++) {
			if (maxProb < condProbs.get(k)) { 
				maxProb = condProbs.get(k);
				maxK = k;
			}
		}
		
		return maxK;
	}
	
	
	@Override
	public String getName() {
		String name = getConfig().getAsString(DUPLICATED_ALG_NAME_FIELD);
		if (name != null && !name.isEmpty())
			return name;
		else
			return "mixrem";
	}

	
	@Override
	public void setName(String name) {
		getConfig().put(DUPLICATED_ALG_NAME_FIELD, name);
	}


	@Override
	public String note() {
		return note + 
			"\n" +
			"Users need to specify the number of components (sub-models) via the attribute \"" + COMP_NUMBER_FIELD + "\"";
	}


	@Override
	public DataConfig createDefaultConfig() {
		DataConfig config = super.createDefaultConfig();
		config.put(COMP_NUMBER_FIELD, COMP_NUMBER_DEFAULT);
		config.put(INITIALIZE_GIVEBACK_FIELD, INITIALIZE_GIVEBACK_DEFAULT);
		config.put(EXECUTE_SELECT_COMP_FIELD, EXECUTE_SELECT_COMP_DEFAULT);
		config.addReadOnly(DUPLICATED_ALG_NAME_FIELD);
		return config;
	}

	
	/**
	 * Randomized sampling the specified data.
	 * @param data the specified data.
	 * @param recordNumber the number of randomized records.
	 * @param giveBack if true, the random record is given back to original sample.
	 * @return Randomized sample the specified data.
	 */
	private static LargeStatistics randomSampling(LargeStatistics data, int recordNumber, boolean giveBack) {
		if (data.getZData().size() == 0 || recordNumber <=0 )
			return null;
		
		List<double[]> xData = Util.newList();
		List<double[]> zData = Util.newList();
		Random rnd = new Random();
		for (int i = 0; i < recordNumber; i++) {
			int N = data.getZData().size();
			if (N == 0)
				break;
			int j = rnd.nextInt(N);
			xData.add(data.getXData().get(j));
			zData.add(data.getZData().get(j));
			
			if (!giveBack) {
				data.getXData().remove(j);
				data.getZData().remove(j);
			}
		}
		
		if (zData.size() == 0)
			return null;
		else
			return new LargeStatistics(xData, zData);
	}


	/**
	 * Getting the fitness criterion of this model given large statistics. This method is now deprecated.
	 * @param stat given large statistics.
	 * @return the fitness criterion of this model given large statistics. Return NaN if any error raises.
	 * @throws RemoteException if any error raises.
	 */
	@SuppressWarnings("unused")
	@Deprecated
	private synchronized double getFitness2(LargeStatistics stat) throws RemoteException {
		@SuppressWarnings("unchecked")
		List<ExchangedParameter> parameters = (List<ExchangedParameter>)getParameter();
		if (stat == null || parameters == null || parameters.size() == 0)
			return Constants.UNUSED;
		
		int N = stat.getZData().size();
		if (N == 0) return Constants.UNUSED;
		double fitness = 0.0;
		for (int i = 0; i < N; i++) {
			double[] xVector = stat.getXData().get(i);
			double[] zVector = stat.getZData().get(i);
			
			List<Double> probs = ExchangedParameter.normalZCondProbs(parameters, xVector, zVector);
			double[] max = MathUtil.findExtremeValue(probs, true);
			if (max != null)
				fitness += max[0];
		}
		
		return fitness / (double)N;
	}

	
}
