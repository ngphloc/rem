package net.hudup.regression.em;

import static net.hudup.regression.AbstractRegression.extractNumber;
import static net.hudup.regression.AbstractRegression.notSatisfy;
import static net.hudup.regression.AbstractRegression.splitIndices;
import static net.hudup.regression.em.RegressionEMImpl.R_CALC_VARIANCE_FIELD;

import java.util.Arrays;
import java.util.List;

import net.hudup.core.Constants;
import net.hudup.core.Util;
import net.hudup.core.alg.Alg;
import net.hudup.core.alg.DuplicatableAlg;
import net.hudup.core.data.AttributeList;
import net.hudup.core.data.DataConfig;
import net.hudup.core.data.Fetcher;
import net.hudup.core.data.Profile;

/**
 * This class implements the semi-mixture regression model.
 * If there are 2 partial models and dual mode is true, this semi-mixture regression model becomes dual regression model.
 *  
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class SemiMixtureRegressionEM extends AbstractMixtureRegressionEM implements DuplicatableAlg {

	
	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Field name of mutual mode.
	 */
	protected final static String MUTUAL_MODE_FIELD = "srem_mutual_mode";
	
	
	/**
	 * Default mutual mode.
	 */
	protected final static boolean MUTUAL_MODE_DEFAULT = false;

	
	/**
	 * Field name of mutual mode.
	 */
	protected final static String UNIFORM_MODE_FIELD = "srem_uniform_mode";
	
	
	/**
	 * Default mutual mode.
	 */
	protected final static boolean UNIFORM_MODE_DEFAULT = false;

	
	/**
	 * Field name of decomposition.
	 */
	protected final static String DECOMPOSE_FIELD = "srem_decompose";
	
	
	/**
	 * Default decomposition.
	 */
	protected final static boolean DECOMPOSE_DEFAULT = true;

	
	/**
	 * Field name of logistic mode.
	 */
	protected final static String LOGISTIC_MODE_FIELD = "srem_logistic_mode";
	
	
	/**
	 * Default logistic mode.
	 */
	protected final static boolean LOGISTIC_MODE_DEFAULT = false;

	
	@Override
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
			RegressionEMImpl rem = createRegressionEM();
			rem.getConfig().put(R_INDICES_FIELD, indicesList.get(i));
			rem.setup(inputSample);
			if(rem.attList != null) // if rem is set up successfully.
				this.rems.add(rem);
		}
		
		if (this.rems.size() == 0) {
			this.rems = null;
			return false;
		}
		else
			return true;
	}

	
	@Override
	protected RegressionEMImpl createRegressionEM() {
		// TODO Auto-generated method stub
		RegressionEMImpl rem = super.createRegressionEM();
		rem.getConfig().put(R_CALC_VARIANCE_FIELD, true);
		return rem;
	}


	/**
	 * Expectation method of this class does not change internal data.
	 */
	@Override
	protected Object expectation(Object currentParameter, Object...info) throws Exception {
		// TODO Auto-generated method stub
		@SuppressWarnings("unchecked")
		List<LargeStatistics> stats = (List<LargeStatistics>)super.expectation(currentParameter, info);
		
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
	protected Object initializeParameter() {
		// TODO Auto-generated method stub
		@SuppressWarnings("unchecked")
		List<ExchangedParameter> parameters = (List<ExchangedParameter>)super.initializeParameter();
		
		for (ExchangedParameter parameter : parameters) {
			parameter.setCoeff(Constants.UNUSED);
			parameter.setZVariance(Constants.UNUSED);
		}
		return parameters;
	}


	@Override
	protected boolean adjustMixtureParameters() throws Exception {
		if (this.rems == null || this.rems.size() == 0)
			return false;
		
		for (RegressionEMImpl rem : this.rems) {
			ExchangedParameter parameter = rem.getExchangedParameter();
			double zVariance = parameter.estimateZVariance(rem.getLargeStatistics());
			parameter.setZVariance(zVariance);

			if (!getConfig().getAsBoolean(LOGISTIC_MODE_FIELD))
				parameter.setCoeff(1.0 / (double)this.rems.size());
		}
		//In uniform mode, all coefficients are 1/K. In logistic mode, coefficients are not used. 
		if (getConfig().getAsBoolean(UNIFORM_MODE_FIELD) || getConfig().getAsBoolean(LOGISTIC_MODE_FIELD))
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
				
				List<Double> condProbs = condZProbs(parameterList, XList, Arrays.asList(new double[] {1, zValue}));
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
	 * Adjusting specified parameters based on specified statistics according to mixture model for many iterations.
	 * This method is replaced by {@link #adjustMixtureParametersOne()} method.
	 * @return true if the adjustment process is successful.
	 * @throws Exception if any error raises.
	 */
	@Deprecated
	protected boolean adjustMixtureParameters2() throws Exception {
		if (this.rems == null || this.rems.size() == 0)
			return false;
		
		for (RegressionEMImpl rem : this.rems) {
			ExchangedParameter parameter = rem.getExchangedParameter();
			double zVariance = parameter.estimateZVariance(rem.getLargeStatistics());
			parameter.setZVariance(zVariance);

			//In logistic mode, coefficients are not used. 
			if (!getConfig().getAsBoolean(LOGISTIC_MODE_FIELD))
				parameter.setCoeff(1.0 / (double)this.rems.size());
		}
		//In uniform mode, all coefficients are 1/K. In logistic mode, coefficients are not used. 
		if (getConfig().getAsBoolean(UNIFORM_MODE_FIELD) || getConfig().getAsBoolean(LOGISTIC_MODE_FIELD))
			return true;
		
		boolean terminated = true;
		int t = 0;
		int maxIteration = getConfig().getAsInt(EM_MAX_ITERATION_FIELD);
		maxIteration = (maxIteration <= 0) ? EM_MAX_ITERATION : maxIteration;
		double threshold = getConfig().getAsReal(EM_EPSILON_FIELD);
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
					
					List<Double> condProbs = condZProbs(parameterList, XList, Arrays.asList(new double[] {1, zValue}));
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

	
	@Override
	public synchronized Object execute(Object input) {
		// TODO Auto-generated method stub
		if (getConfig().getAsBoolean(LOGISTIC_MODE_FIELD)) { // Logistic mode does not use probability
			List<Double> zValues = Util.newList(this.rems.size());
			List<Double> expProbs = Util.newList(this.rems.size());
			double expProbsSum = 0;
			for (int k = 0; k < this.rems.size(); k++) {
				RegressionEMImpl rem = this.rems.get(k);
				double zValue = extractNumber(rem.execute(input));
				if (!Util.isUsed(zValue))
					return null;
				
				zValues.add(zValue);
				
				ExchangedParameter parameter = rem.getExchangedParameter();
				double prob = ExchangedParameter.normalPDF(zValue, 
						parameter.mean(rem.extractRegressors(input)),
						parameter.getZVariance());
				double weight = Math.exp(prob);
				expProbs.add(weight);
				expProbsSum += weight;
			}

			double result = 0;
			for (int k = 0; k < this.rems.size(); k++) {
				result += (expProbs.get(k) / expProbsSum) * zValues.get(k); 
			}
			
			return result;
		}
		else
			return super.execute(input);
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
		config.put(MUTUAL_MODE_FIELD, MUTUAL_MODE_DEFAULT);
		config.put(UNIFORM_MODE_FIELD, UNIFORM_MODE_DEFAULT);
		config.put(DECOMPOSE_FIELD, DECOMPOSE_DEFAULT);
		config.put(LOGISTIC_MODE_FIELD, LOGISTIC_MODE_DEFAULT);
		
		config.addReadOnly(DUPLICATED_ALG_NAME_FIELD);
		return config;
	}

	
}
