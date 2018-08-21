package net.hudup.regression.em;

import static net.hudup.regression.AbstractRegression.extractNumber;
import static net.hudup.regression.AbstractRegression.splitIndices;

import java.util.List;

import net.hudup.core.Util;
import net.hudup.core.alg.Alg;
import net.hudup.core.alg.DuplicatableAlg;
import net.hudup.core.data.AttributeList;
import net.hudup.core.data.DataConfig;
import net.hudup.core.data.Fetcher;
import net.hudup.core.data.Profile;
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
	 * List of internal regression model as parameter.
	 */
	protected List<RegressionEMImpl> rems = Util.newList();

	
	@Override
	@SuppressWarnings("unchecked")
	public Object learn(Object...info) throws Exception {
		// TODO Auto-generated method stub
		if (!prepareInternalData(this.sample)) {
			clearInternalContent();
			return null;
		}
		
		List<ExchangedParameter> parameters = (List<ExchangedParameter>)super.learn();
		if (parameters == null || parameters.size() != this.rems.size()) {
			clearInternalContent();
			return null;
		}
		
		adjustMixtureParametersOne(parameters, (List<LargeStatistics>)this.getStatistics());
		return parameters;
	}
	
	
	@Override
	public synchronized void unsetup() {
		// TODO Auto-generated method stub
		super.unsetup();
		for (RegressionEMImpl rem : this.rems) {
			if (rem != null)
				rem.unsetup();
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
			
			StringBuffer indices = new StringBuffer();
			for (int i = 0; i < attList.size(); i++) {
				if (i > 0)
					indices.append(", ");
				indices.append(i);
			}
			indicesList.add(indices.toString());
			
			if (attList.size() > 2) {
				for (int i = 0; i < attList.size() - 1; i++) {
					indicesList.add(i + ", " + (attList.size() - 1));
				}
			}
		}
			
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
		
		return (this.rems.size() == 0 ? false : true);
	}
	
	
	/**
	 * Clear all internal data.
	 */
	protected void clearInternalContent() {
		this.currentIteration = 0;
		this.currentParameter = this.estimatedParameter = null;
		for (RegressionEMImpl rem : this.rems) {
			if (rem != null)
				rem.clearInternalContent();
		}
		this.rems.clear();
	}

	
	@Override
	protected Object expectation(Object currentParameter, Object...info) throws Exception {
		// TODO Auto-generated method stub
		@SuppressWarnings("unchecked")
		List<ExchangedParameter> parameters = (List<ExchangedParameter>)currentParameter;
		if (parameters.size() == 0)
			return null;
		
		List<LargeStatistics> stats = Util.newList(parameters.size());
		for (int i = 0; i < parameters.size(); i++) {
			ExchangedParameter parameter = parameters.get(i);
			RegressionEMImpl rem = this.rems.get(i);
			if (parameter == null || rem == null)
				continue;
			
			LargeStatistics stat = (LargeStatistics)rem.expectation(parameter);
			if (stat != null)
				stats.add(stat);
		}
		if (stats.size() != parameters.size()) {
			logger.error("Some regression models are failed in calculating expectation");
			return null;
		}
		
		//Supporting mutual mode. If there are two components, it is dual mode.
		//Mutual mode is useful in some cases.
		if (getConfig().getAsBoolean(MUTUAL_MODE_FIELD)) {
			//Retrieving same-length Z statistics
			int N = stats.get(0).getZStatistic().size();
			for (LargeStatistics stat : stats) {
				if (stat.zData.size() != N)
					return null;
			}
			
			//Calculating average value of Z in mutual mode.
			for (int i = 0; i < N; i++) {
				double mean = 0;
				for (int k = 0; k < stats.size(); k++) {
					mean += parameters.get(k).getCoeff() * stats.get(k).getZStatisticMean(); 
				}
				for (int k = 0; k < stats.size(); k++) {
					LargeStatistics stat = stats.get(k);
					stat.zData.get(i)[1] = mean;
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
		if (stats.size() == 0)
			return null;

		List<ExchangedParameter> parameters = Util.newList(stats.size());
		for (int i = 0; i < stats.size(); i++) {
			LargeStatistics stat = stats.get(i);
			RegressionEMImpl rem = this.rems.get(i);
			if (stat == null || rem == null)
				continue;
			
			ExchangedParameter parameter = (ExchangedParameter)rem.maximization(stat);
			if (parameter != null) {
				rem.setParameter(parameter, this.getCurrentIteration());
				parameters.add(parameter);
			}
		}
		if (parameters.size() != stats.size()) {
			logger.error("Some regression models are failed in maximization");
			return null;
		}
		
		return parameters;
	}

	
	/**
	 * Adjusting specified parameters based on specified statistics according to mixture model in one iteration.
	 * This method does not need a loop because both mean and variance were optimized in REM process and so the probabilities of components will be optimized in only one time.
	 * @param parametersInOut specified parameters. They are also output parameters.
	 * @param stats specified statistics.
	 * @return true if the adjustment process is successful.
	 */
	protected boolean adjustMixtureParametersOne(List<ExchangedParameter> parametersInOut, List<LargeStatistics> stats) {
		if (parametersInOut == null || stats == null || parametersInOut.size() == 0 || stats.size() == 0 || parametersInOut.size() != stats.size())
			return false;
		
		List<double[]> zData = this.rems.get(0).data.zData; //All models have the same original Z variables.
		for (int k = 0; k < parametersInOut.size(); k++) {
			ExchangedParameter parameter = parametersInOut.get(k);
			double condProbSum = 0;
			int N = 0;
			for (int i = 0; i < zData.size(); i++) {
				double zStatistic = zData.get(i)[1];
				if (Util.isUsed(zStatistic)) {
					double[] condProbs = condProbs(zStatistic, parametersInOut);
					condProbSum += condProbs[k];
					N++;
				}
			}
			if (condProbSum == 0)
				logger.warn("#adjustMixtureParameters: zero sum of conditional probabilities in " + k + "th model");
			double coeff = condProbSum / (double)N;
			parameter.setCoeff(coeff);
		}
		
		return true;
	}
	
	
	/**
	 * Adjusting specified parameters based on specified statistics according to mixture model.
	 * This method is deprecated because coefficients, mean and variance of a regression model is already optimized by EM process.
	 * It is over-fitting if continue to optimize by mixture process. However, mixture process is reduced into one iteration to estimate the probabilities of partial components.
	 * Please see the method {@link #adjustMixtureParametersOne(List, List)}.
	 * @param parametersInOut specified parameters. They are also output parameters.
	 * @param stats specified statistics.
	 * @return true if the adjustment process is successful.
	 */
	@Deprecated
	protected boolean adjustMixtureParameters(List<ExchangedParameter> parametersInOut, List<LargeStatistics> stats) {
		if (parametersInOut == null || stats == null || parametersInOut.size() == 0 || stats.size() == 0 || parametersInOut.size() != stats.size())
			return false;
		double threshold = getConfig().getAsReal(EM_EPSILON_FIELD);
		
		for (int k = 0; k < parametersInOut.size(); k++) {
			ExchangedParameter parameter = parametersInOut.get(k);
			double mean = stats.get(k).getZStatisticMean();
			double variance = stats.get(k).getZStatisticBiasedVariance();
			parameter.setMean(mean);
			parameter.setVariance(variance);
		}
		
		boolean terminated = true;
		int t = 0;
		int maxIteration = getConfig().getAsInt(EM_MAX_ITERATION_FIELD);
		maxIteration = (maxIteration <= 0) ? EM_MAX_ITERATION : maxIteration;
		do {
			terminated = true;
			t++;
			for (int k = 0; k < parametersInOut.size(); k++) {
				ExchangedParameter parameter = parametersInOut.get(k); 
				List<double[]> ZStatistic = stats.get(k).zData;
				int N = ZStatistic.size();
				
				double condProbSum = 0;
				double zSum = 0;
				List<double[]> condProbsList = Util.newList(N);
				for (int i = 0; i < N; i++) {
					double[] condProbs = condProbs(ZStatistic.get(i)[1], parametersInOut);
					condProbsList.add(condProbs);
					
					condProbSum += condProbs[k];
					zSum += condProbs[k] * ZStatistic.get(i)[1];
				}
				if (condProbSum == 0)
					logger.warn("#adjustMixtureParameters: zero sum of conditional probabilities in " + k + "th model");
				
				//Estimating coefficient
				double coeff = condProbSum / (double)N;
				if (Util.isUsed(parameter.getCoeff())
						&& Math.abs(coeff - parameter.getCoeff()) > threshold * Math.abs(parameter.getCoeff()))
					terminated = terminated && false;
				parameter.setCoeff(coeff);
				
				//Estimating mean
				double mean = zSum / condProbSum;
				if (Util.isUsed(parameter.getMean())
						&& Math.abs(mean - parameter.getMean()) > threshold * Math.abs(parameter.getMean()))
					terminated = terminated && false;
				parameter.setMean(mean);
	
				//Estimating variance
				double zDevSum = 0;
				for (int i = 0; i < N; i++) {
					double[] condProbs = condProbsList.get(i);
					double d = ZStatistic.get(i)[1] - mean;
					zDevSum += condProbs[k] * (d*d);
				}
				double variance = zDevSum / condProbSum;
				if (Util.isUsed(parameter.getVariance())
						&& Math.abs(variance - parameter.getVariance()) > threshold * Math.abs(parameter.getVariance()))
					terminated = terminated && false;
				parameter.setVariance(variance);
				
				if (variance == 0)
					logger.warn("#adjustMixtureParameters: Variance of the " + k + "th model is 0");
			}
		} while (!terminated && t < maxIteration);
		
		return true;
	}
	
	
	/**
	 * Calculating the condition probabilities of the specified parameters given response value (Z).
	 * Inherited class can re-define this method. In current version, only normal probability density function is used.
	 * @param z given response value (Z).
	 * @param parameters arrays of parameters.
	 * @return condition probabilities of the specified parameters given response value (Z).
	 */
	protected double[] condProbs(double z, List<ExchangedParameter> parameters) {
		return ExchangedParameter.normalCondProbs(z, parameters.toArray(new ExchangedParameter[] {}));
	}

	
	@Override
	protected Object initializeParameter() {
		// TODO Auto-generated method stub
		if (this.rems.size() == 0)
			return null;
		List<ExchangedParameter> parameters = Util.newList(this.rems.size());
			
		for (RegressionEMImpl rem : this.rems) {
			if (rem == null)
				continue;
			
			ExchangedParameter parameter = (ExchangedParameter)rem.initializeParameter();
			if (parameter != null) {
				rem.setParameter(parameter, 0);
				parameters.add(parameter);
			}
		}
		if (parameters.size() != this.rems.size()) {
			logger.error("Some regression models are failed in initialization");
			return null;
		}
		
		for (int k = 0; k < parameters.size(); k++) {
			ExchangedParameter parameter = parameters.get(k);
			LargeStatistics zData = this.rems.get(k).getData();
			
			double coeff = (1.0 / (double)parameters.size());
			parameter.setCoeff(coeff);
			
			double mean = zData.getZStatisticMean();
			parameter.setMean(mean);
			
			double variance = zData.getZStatisticBiasedVariance();
			parameter.setVariance(variance);
		}
		
		return parameters;
	}

	
	@Override
	protected boolean terminatedCondition(Object estimatedParameter, Object currentParameter, Object previousParameter, Object... info) {
		// TODO Auto-generated method stub
		@SuppressWarnings("unchecked")
		List<ExchangedParameter> exEstimatedParameters = (List<ExchangedParameter>)estimatedParameter;
		@SuppressWarnings("unchecked")
		List<ExchangedParameter> exCurrentParameters = (List<ExchangedParameter>)currentParameter;
		@SuppressWarnings("unchecked")
		List<ExchangedParameter> exPreviousParameters = (List<ExchangedParameter>)previousParameter;

		boolean terminated = true;
		for (int i = 0; i < this.rems.size(); i++) {
			RegressionEMImpl rem = this.rems.get(i);
			ExchangedParameter exEstimatedParameter = exEstimatedParameters.get(i);
			ExchangedParameter exCurrentParameter = exCurrentParameters.get(i);
			ExchangedParameter exPreviousParameter = exPreviousParameters != null ? exPreviousParameters.get(i) : null;
			
			if (rem == null || exEstimatedParameter == null || exCurrentParameter == null)
				continue;
			
			terminated = terminated && rem.terminatedCondition(exEstimatedParameter, exCurrentParameter, exPreviousParameter, info);
			if (!terminated)
				return false;
		}
		
		return terminated;
	}

	
	@Override
	public synchronized Object execute(Object input) {
		// TODO Auto-generated method stub
		double result = 0;
		for (int i = 0; i < this.rems.size(); i++) {
			RegressionEMImpl rem = this.rems.get(i);
			ExchangedParameter parameter = (ExchangedParameter)rem.getParameter();
			
			double value = extractNumber(rem.execute(input));
			if (Util.isUsed(value))
				result += parameter.getCoeff() * value;
			else
				return null;
		}
		return result;
	}

	
	@Override
	public synchronized String parameterToShownText(Object parameter, Object... info) {
		// TODO Auto-generated method stub
		if (parameter == null || !(parameter instanceof List<?>))
			return "";
		
		StringBuffer buffer = new StringBuffer();
		@SuppressWarnings("unchecked")
		List<ExchangedParameter> exParameters = (List<ExchangedParameter>)parameter;
		for (int i = 0; i < exParameters.size(); i++) {
			if (i > 0)
				buffer.append(", ");
			String text = "";
			if (this.rems.get(i) != null)
				text = this.rems.get(i).parameterToShownText(exParameters.get(i), info);
			buffer.append("{" + text + "}");
		}
		
		return buffer.toString();
	}

	
	@Override
	public synchronized String getDescription() {
		// TODO Auto-generated method stub
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
		SemiMixtureRegressionEM mixREM = new SemiMixtureRegressionEM();
		mixREM.getConfig().putAll((DataConfig)this.getConfig().clone());
		return mixREM;
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
		config.put(R_INDICES_FIELD, R_INDICES_FIELD_DEFAULT);
		config.put(MUTUAL_MODE_FIELD, MUTUAL_MODE_DEFAULT);
		
		config.addReadOnly(DUPLICATED_ALG_NAME_FIELD);
		return config;
	}

	
	@Override
	public synchronized Object extractResponse(Profile profile) {
		// TODO Auto-generated method stub
		double mean = 0;
		for (RegressionEMImpl rem : this.rems) {
			if (rem == null)
				return null;
			
			double value = AbstractRegression.extractNumber(rem.extractResponse(profile));
			if (!Util.isUsed(value))
				return null;
			
			mean += value * ((ExchangedParameter)rem.getParameter()).getCoeff();
		}
		return mean;
	}


}
