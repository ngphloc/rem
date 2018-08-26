package net.hudup.regression.em;

import static net.hudup.regression.AbstractRegression.*;
import static net.hudup.regression.AbstractRegression.splitIndices;

import java.util.List;

import net.hudup.core.Util;
import net.hudup.core.alg.Alg;
import net.hudup.core.alg.DuplicatableAlg;
import net.hudup.core.data.AttributeList;
import net.hudup.core.data.DataConfig;
import net.hudup.core.data.Fetcher;
import net.hudup.core.data.Profile;
import net.hudup.core.logistic.MathUtil;
import net.hudup.em.ExponentialEM;
import net.hudup.regression.AbstractRegression;
import net.hudup.regression.Regression;

/**
 * This class implements expectation maximization (EM) algorithm for mixture regression models.
 * If there are 2 partial models (by default), the mixture regression model becomes dual regression model.
 *  
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class SemiMixtureRegressionEM extends ExponentialEM implements Regression, DuplicatableAlg {

	
	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Field name of mutual mode.
	 */
	protected final static String MUTUAL_MODE_FIELD = "mixrem_mutual_mode";
	
	
	/**
	 * Default mutual mode.
	 */
	protected final static boolean MUTUAL_MODE_DEFAULT = false;

	
	/**
	 * Field name of mutual mode.
	 */
	protected final static String UNIFORM_MODE_FIELD = "mixrem_uniform_mode";
	
	
	/**
	 * Default mutual mode.
	 */
	protected final static boolean UNIFORM_MODE_DEFAULT = false;

	
	/**
	 * List of internal regression model as parameter.
	 */
	protected List<RegressionEMImpl> rems = null;

	
	@Override
	public Object learn(Object...info) throws Exception {
		// TODO Auto-generated method stub
		if (!prepareInternalData(this.sample)) {
			clearInternalContent();
			return null;
		}
		
		if (super.learn() == null) {
			clearInternalContent();
			return null;
		}
		
		adjustMixtureParametersOne();
		
		return this.rems;
	}
	
	
	@Override
	public synchronized void unsetup() {
		// TODO Auto-generated method stub
		super.unsetup();
		if (this.rems != null) {
			for (RegressionEMImpl rem : this.rems) {
				if (rem != null)
					rem.unsetup();
			}
		}
	}

	
	/**
	 * Preparing data.
	 * @param inputSample specified sample.
	 * @return true if data preparation is successful.
	 * @throws Exception if any error raises.
	 */
	protected boolean prepareInternalData(Fetcher<Profile> inputSample) throws Exception {
		clearInternalContent();
		DataConfig thisConfig = this.getConfig();
		
		List<String> indicesList = splitIndices(thisConfig.getAsString(R_INDICES_FIELD));
		if (indicesList.size() == 0) {
			AttributeList attList = getSampleAttributeList(inputSample);
			if (attList.size() < 2)
				return false;
			
//			StringBuffer indices = new StringBuffer();
//			for (int i = 1; i <= attList.size(); i++) {
//				if (i > 1)
//					indices.append(", ");
//				indices.append(i);
//			}
//			indicesList.add(indices.toString());
//			if (attList.size() > 2) {
//				for (int i = 1; i < attList.size(); i++) {
//					indicesList.add(i + ", " + attList.size());
//				}
//			}
			
			for (int i = 1; i < attList.size(); i++) {// For fair test
				indicesList.add(i + ", " + attList.size());
			}
		}
		
		this.rems = Util.newList(indicesList.size());
		for (int i = 0; i < indicesList.size(); i++) {
			RegressionEMImpl rem = new RegressionEMImpl() {

				/**
				 * Serial version UID for serializable class.
				 */
				private static final long serialVersionUID = 1L;

				@Override
				public synchronized Object learn(Object...info) throws Exception {
					// TODO Auto-generated method stub
					boolean prepared = prepareInternalData(inputSample);
					if (prepared)
						return prepared;
					else
						return null;
				}

			};
			
			DataConfig config = rem.getConfig();
			config.put(R_INDICES_FIELD, indicesList.get(i));
			rem.setup(inputSample);
			if(rem.attList != null) // if rem1 is set up successfully.
				this.rems.add(rem);
		}
		
		if (this.rems.size() == 0) {
			this.rems = null;
			return false;
		}
		else
			return true;
	}
	
	
	/**
	 * Clear all internal data.
	 */
	protected void clearInternalContent() {
		this.currentIteration = 0;
		this.estimatedParameter = this.currentParameter = this.previousParameter = null;
		
		if (this.rems != null) {
			for (RegressionEMImpl rem : this.rems) {
				if (rem != null)
					rem.clearInternalContent();
			}
			this.rems.clear();
			this.rems = null;
		}
	}

	
	@Override
	protected Object expectation(Object currentParameter, Object...info) throws Exception {
		// TODO Auto-generated method stub
		@SuppressWarnings("unchecked")
		List<ExchangedParameter> parameters = (List<ExchangedParameter>)currentParameter;
		List<LargeStatistics> stats = Util.newList(this.rems.size());
		for (int k = 0; k < this.rems.size(); k++) {
			RegressionEMImpl rem = this.rems.get(k);
			ExchangedParameter parameter = parameters.get(k);
//			if (rem.terminatedCondition(rem.getEstimatedParameter(), rem.getCurrentParameter(), rem.getPreviousParameter(), info)
//				&& rem.getLargeStatistics() != null && rem.getLargeStatistics() != rem.getData())
//				continue;
				
			LargeStatistics stat = (LargeStatistics)rem.expectation(parameter);
			rem.setStatistics(stat);
			stats.add(stat);
			
			if (stat == null)
				logger.error("Some regression models are failed in expectation");
		}
		
		//Supporting mutual mode. If there are two components, it is dual mode.
		//Mutual mode is useful in some cases.
		if (getConfig().getAsBoolean(MUTUAL_MODE_FIELD)) {
			//Retrieving same-length Z statistics
			LargeStatistics stat0 = this.rems.get(0).getLargeStatistics();
			int N = stat0.getZData().size();
			for (RegressionEMImpl rem : this.rems) {
				LargeStatistics stat = rem.getLargeStatistics();
				if (stat.getZData().size() != N)
					return null;
			}
			
			//Calculating average value of Z in mutual mode.
			for (int i = 0; i < N; i++) {
				double mean0 = 0;
				double coeffSum = 0;
				for (RegressionEMImpl rem : this.rems) {
					ExchangedParameter parameter = rem.getExchangedParameter();
					LargeStatistics stat = rem.getLargeStatistics();
					double mean = parameter.mean(stat.getXData().get(i));
					double coeff = parameter.getCoeff();
					coeff = Util.isUsed(coeff) ? coeff : 1;
					mean0 += coeff * mean;
					coeffSum += coeff;
				}
				mean0 = mean0 / coeffSum;
				
				for (RegressionEMImpl rem : this.rems) {
					LargeStatistics stat = rem.getLargeStatistics();
					stat.getZData().get(i)[1] = mean0;
				}
			}
		}
		
		return stats;
	}

	
	@Override
	protected Object maximization(Object currentStatistic, Object...info) throws Exception {
		// TODO Auto-generated method stub
		@SuppressWarnings("unchecked")
		List<LargeStatistics> stats = (List<LargeStatistics>)currentStatistic;
		List<ExchangedParameter> parameters = Util.newList(this.rems.size());
		for (int k = 0; k < this.rems.size(); k++) {
			RegressionEMImpl rem = this.rems.get(k);
			LargeStatistics stat = stats.get(k);
//			if (rem.terminatedCondition(rem.getEstimatedParameter(), rem.getCurrentParameter(), rem.getPreviousParameter(), info))
//				continue;

			ExchangedParameter parameter = (ExchangedParameter)rem.maximization(stat);
			rem.setEstimatedParameter(parameter);
			parameters.add(parameter);
			
			if (parameter == null)
				logger.error("Some regression models are failed in maximization");
		}
		
		return parameters;
	}

	
	@Override
	protected Object initializeParameter() {
		// TODO Auto-generated method stub
		List<ExchangedParameter> parameters = Util.newList(this.rems.size());
		for (int k = 0; k < this.rems.size(); k++) {
			RegressionEMImpl rem = this.rems.get(k);
			ExchangedParameter parameter = (ExchangedParameter)rem.initializeParameter();
			parameters.add(parameter);
			
			rem.setEstimatedParameter(parameter);
			rem.setCurrentParameter(parameter);
			rem.setPreviousParameter(null);
			rem.setStatistics(null);
			rem.setCurrentIteration(0);
			
			if (parameter == null)
				logger.error("Some regression models are failed in initialization");
		}
		return parameters;
	}

	
	@Override
	public synchronized void permuteNotify() {
		// TODO Auto-generated method stub
		super.permuteNotify();
		@SuppressWarnings("unchecked")
		List<ExchangedParameter> estimatedParameters = (List<ExchangedParameter>)getEstimatedParameter();
		@SuppressWarnings("unchecked")
		List<ExchangedParameter> currentParameters = (List<ExchangedParameter>)getCurrentParameter();
		@SuppressWarnings("unchecked")
		List<ExchangedParameter> previousParameters = (List<ExchangedParameter>)getPreviousParameter();
		
		for (int k = 0; k < this.rems.size(); k++) {
			RegressionEMImpl rem = this.rems.get(k);
			ExchangedParameter estimatedParameter = estimatedParameters.get(k);
			ExchangedParameter currentParameter = currentParameters.get(k);
			ExchangedParameter previousParameter = previousParameters != null ? previousParameters.get(k) : null;
			
			rem.setEstimatedParameter(estimatedParameter);
			rem.setCurrentParameter(currentParameter);
			rem.setPreviousParameter(previousParameter);
			rem.setCurrentIteration(this.getCurrentIteration());
		}
	}


	@Override
	public synchronized void finishNotify() {
		// TODO Auto-generated method stub
		super.finishNotify();
		@SuppressWarnings("unchecked")
		List<ExchangedParameter> estimatedParameters = (List<ExchangedParameter>)getEstimatedParameter();
		@SuppressWarnings("unchecked")
		List<ExchangedParameter> currentParameters = (List<ExchangedParameter>)getCurrentParameter();
		
		for (int k = 0; k < this.rems.size(); k++) {
			RegressionEMImpl rem = this.rems.get(k);
			ExchangedParameter estimatedParameter = estimatedParameters.get(k);
			ExchangedParameter currentParameter = currentParameters.get(k);
			
			rem.setEstimatedParameter(estimatedParameter);
			rem.setCurrentParameter(currentParameter);
			rem.setCurrentIteration(this.getCurrentIteration());
		}
	}


	/**
	 * Adjusting specified parameters based on specified statistics according to mixture model in one iteration.
	 * This method does not need a loop because both mean and variance were optimized in REM process and so the probabilities of components will be optimized in only one time.
	 * @return true if the adjustment process is successful.
	 * @throws Exception if any error raises.
	 */
	protected boolean adjustMixtureParametersOne() throws Exception {
		if (this.rems == null || this.rems.size() == 0)
			return false;
		
		for (RegressionEMImpl rem : this.rems) {
			ExchangedParameter parameter = rem.getExchangedParameter();
			parameter.setCoeff(1.0 / (double)this.rems.size());
			double zVariance = parameter.estimateZVariance(rem.getLargeStatistics());
			parameter.setZVariance(zVariance);
		}
		if (getConfig().getAsBoolean(UNIFORM_MODE_FIELD)) //In uniform mode, all coefficients are 1/K
			return true;
		
		List<ExchangedParameter> parameterList = Util.newList(this.rems.size());
		for (RegressionEMImpl rem : this.rems) {
			ExchangedParameter parameter = rem.getExchangedParameter();
			parameterList.add((ExchangedParameter)parameter.clone());
		}
		
		this.currentIteration++;
		for (int k = 0; k < this.rems.size(); k++) {
			RegressionEMImpl rem = this.rems.get(k);
			ExchangedParameter parameter = rem.getExchangedParameter();
			
			double condProbSum = 0;
			int N = 0;
			List<double[]> zData = rem.getData().getZData(); //By default, all models have the same original Z variables.
			for (int i = 0; i < zData.size(); i++) {
				double zValue = zData.get(i)[1];
				if (!Util.isUsed(zValue))
					continue;
				
				List<double[]> XList = Util.newList(this.rems.size());
				for (RegressionEMImpl rem2 : this.rems) {
					XList.add(rem2.getLargeStatistics().getXData().get(i));
				}
				
				List<Double> condProbs = condZProbs(parameterList, XList, zValue);
				condProbSum += condProbs.get(k);
				N++;
			}
			if (condProbSum == 0)
				logger.warn("#adjustMixtureParameters: zero sum of conditional probabilities in " + k + "th model");
			
			//Estimating coefficient
			double coeff = condProbSum / (double)N;
			parameter.setCoeff(coeff);
		}
		
		return true;
	}
	
	
	/**
	 * Adjusting specified parameters based on specified statistics according to mixture model in one iteration.
	 * @return true if the adjustment process is successful.
	 * @throws Exception if any error raises.
	 */
	@Deprecated
	protected boolean adjustMixtureParameters() throws Exception {
		if (this.rems == null || this.rems.size() == 0)
			return false;
		double threshold = getConfig().getAsReal(EM_EPSILON_FIELD);
		
		for (RegressionEMImpl rem : this.rems) {
			ExchangedParameter parameter = rem.getExchangedParameter();
			parameter.setCoeff(1.0 / (double)this.rems.size());
			double zVariance = parameter.estimateZVariance(rem.getLargeStatistics());
			parameter.setZVariance(zVariance);
		}
		if (getConfig().getAsBoolean(UNIFORM_MODE_FIELD)) //In uniform mode, all coefficients are 1/K
			return true;
		
		boolean terminated = true;
		int t = 0;
		int maxIteration = getConfig().getAsInt(EM_MAX_ITERATION_FIELD);
		maxIteration = (maxIteration <= 0) ? EM_MAX_ITERATION : maxIteration;
		do {
			terminated = true;
			t++;
			this.currentIteration++;
			
			List<ExchangedParameter> parameterList = Util.newList(this.rems.size());
			for (RegressionEMImpl rem : this.rems) {
				ExchangedParameter parameter = rem.getExchangedParameter();
				parameterList.add((ExchangedParameter)parameter.clone());
			}
			
			for (int k = 0; k < this.rems.size(); k++) {
				RegressionEMImpl rem = this.rems.get(k);
				ExchangedParameter parameter = rem.getExchangedParameter();
				
				double condProbSum = 0;
				int N = 0;
				List<double[]> zData = rem.getData().getZData(); //By default, all models have the same original Z variables.
				//double zSum = 0;
				List<List<Double>> condProbsList = Util.newList(N);
				for (int i = 0; i < zData.size(); i++) {
					double zValue = zData.get(i)[1];
					if (!Util.isUsed(zValue))
						continue;
					
					List<double[]> XList = Util.newList(this.rems.size());
					for (RegressionEMImpl rem2 : this.rems) {
						XList.add(rem2.getLargeStatistics().getXData().get(i));
					}
					
					List<Double> condProbs = condZProbs(parameterList, XList, zValue);
					condProbsList.add(condProbs);
					
					condProbSum += condProbs.get(k);
					//zSum += condProbs.get(k) * zValue;
					N++;
				}
				if (condProbSum == 0)
					logger.warn("#adjustMixtureParameters: zero sum of conditional probabilities in " + k + "th model");
				
				//Estimating coefficient
				double coeff = condProbSum / (double)N;
				if (notSatisfy(coeff, parameter.getCoeff(), threshold))
					terminated = terminated && false;
				parameter.setCoeff(coeff);
				
//				//Estimating mean
//				double mean = zSum / condProbSum;
//				if (notSatisfy(mean, parameter.getMean(), threshold))
//					terminated = terminated && false;
//				parameter.setMean(mean);
//				
//				//Estimating variance
//				double zDevSum = 0;
//				for (int i = 0; i < zData.size(); i++) {
//					double zValue = zData.get(i)[1];
//					if (!Util.isUsed(zValue))
//						continue;
//
//					List<Double> condProbs = condProbsList.get(i);
//					double d = zValue - mean;
//					zDevSum += condProbs.get(k) * (d*d);
//				}
//				double variance = zDevSum / condProbSum;
//				if (notSatisfy(variance, parameter.getVariance(), threshold))
//					terminated = terminated && false;
//				parameter.setVariance(variance);
//				if (variance == 0)
//					logger.warn("#adjustMixtureParameters: Variance of the " + k + "th model is 0");
			}
			
		} while (!terminated && t < maxIteration);
		
		return true;
	}

	
	/**
	 * Calculating the condition probabilities of the specified parameters given regressor values X and response value Z.
	 * Inherited class can re-define this method. In current version, only normal probability density function is used.
	 * @param parameterList list of current parameters.
	 * @param XList regressor values X
	 * @param zValue given response value Z.
	 * @return condition probabilities of the specified parameters given regressor values X and response value Z.
	 */
	protected List<Double> condZProbs(List<ExchangedParameter> parameterList, List<double[]> XList, double zValue) {
		if (parameterList.size() == 0)
			return Util.newList();
		
		return ExchangedParameter.normalZCondProbs(parameterList, XList, zValue);
	}

	
	@Override
	protected boolean terminatedCondition(Object estimatedParameter, Object currentParameter, Object previousParameter, Object... info) {
		// TODO Auto-generated method stub
		@SuppressWarnings("unchecked")
		List<ExchangedParameter> estimatedParameters = (List<ExchangedParameter>)estimatedParameter;
		@SuppressWarnings("unchecked")
		List<ExchangedParameter> currentParameters = (List<ExchangedParameter>)currentParameter;
		@SuppressWarnings("unchecked")
		List<ExchangedParameter> previousParameters = (List<ExchangedParameter>)previousParameter;
		
		boolean terminated = true;
		for (int k = 0; k < this.rems.size(); k++) {
			RegressionEMImpl rem = this.rems.get(k);
			ExchangedParameter pParameter = previousParameters != null ? previousParameters.get(k) : null;
			ExchangedParameter cParameter = currentParameters.get(k);
			ExchangedParameter eParameter = estimatedParameters.get(k);
			
			if (eParameter == null && cParameter == null)
				continue;
			else if (eParameter == null || cParameter == null)
				return false;
			
			terminated = terminated && rem.terminatedCondition(eParameter, cParameter, pParameter, info);
			if (!terminated)
				return false;
		}
		
		return terminated;
	}

	
	@Override
	public synchronized Object execute(Object input) {
		// TODO Auto-generated method stub
		if (this.rems == null || this.rems.size() == 0)
			return null;
		
		double result = 0;
		for (RegressionEMImpl rem : this.rems) {
			ExchangedParameter parameter = (ExchangedParameter)rem.getParameter();
			
			double value = extractNumber(rem.execute(input));
			if (Util.isUsed(value))
				result += parameter.getCoeff() * value;
			else
				return null;
		}
		return result;
	}

	
	/**
	 * Executing this algorithm by arbitrary input parameter.
	 * @param input arbitrary input parameter.
	 * @return result of execution. Return null if execution is failed.
	 */
	public Object executeIntel(Object...input) {
		return execute(input);
	}

	
	@Override
	public synchronized String parameterToShownText(Object parameter, Object... info) {
		// TODO Auto-generated method stub
		if (parameter == null || !(parameter instanceof List<?>))
			return "";
		
		@SuppressWarnings("unchecked")
		List<ExchangedParameter> parameters = (List<ExchangedParameter>)parameter;
		StringBuffer buffer = new StringBuffer();
		for (int k = 0; k < rems.size(); k++) {
			if (k > 0)
				buffer.append(", ");
			String text = rems.get(k).parameterToShownText(parameters.get(k), info);
			buffer.append("{" + text + "}");
		}
		buffer.append(": ");
		buffer.append("t=" + MathUtil.format(getCurrentIteration()));
		
		return buffer.toString();
	}

	
	@Override
	public synchronized String getDescription() {
		// TODO Auto-generated method stub
		if (this.rems == null)
			return "";

		StringBuffer buffer = new StringBuffer();
		for (int i = 0; i < this.rems.size(); i++) {
			Regression regression = this.rems.get(i);
			if (i > 0)
				buffer.append(", ");
			String text = "";
			if (regression != null)
				text = regression.getDescription();
			buffer.append("{" + text + "}");
		}
		buffer.append(": ");
		buffer.append("t=" + getCurrentIteration());
		
		return buffer.toString();
	}

	
	@Override
	public String getName() {
		// TODO Auto-generated method stub
		String name = getConfig().getAsString(DUPLICATED_ALG_NAME_FIELD);
		if (name != null && !name.isEmpty())
			return name;
		else
			return "semi_mixrem";
	}

	
	@Override
	public Alg newInstance() {
		// TODO Auto-generated method stub
		SemiMixtureRegressionEM semiMixREM = new SemiMixtureRegressionEM();
		semiMixREM.getConfig().putAll((DataConfig)this.getConfig().clone());
		return semiMixREM;
	}

	
	@Override
	public void setName(String name) {
		// TODO Auto-generated method stub
		getConfig().put(DUPLICATED_ALG_NAME_FIELD, name);
	}

	
	@Override
	public DataConfig createDefaultConfig() {
		// TODO Auto-generated method stub
		DataConfig config = super.createDefaultConfig();
		config.put(R_INDICES_FIELD, R_INDICES_DEFAULT);
		config.put(MUTUAL_MODE_FIELD, MUTUAL_MODE_DEFAULT);
		config.put(UNIFORM_MODE_FIELD, UNIFORM_MODE_DEFAULT);
		
		config.addReadOnly(DUPLICATED_ALG_NAME_FIELD);
		return config;
	}

	
	@Override
	public synchronized Object extractResponse(Object input) {
		// TODO Auto-generated method stub
		double mean = 0;
		for (RegressionEMImpl rem : this.rems) {
			if (rem == null)
				return null;
			
			double value = AbstractRegression.extractNumber(rem.extractResponse(input));
			if (!Util.isUsed(value))
				return null;
			
			mean += value * ((ExchangedParameter)rem.getParameter()).getCoeff();
		}
		return mean;
	}


}
