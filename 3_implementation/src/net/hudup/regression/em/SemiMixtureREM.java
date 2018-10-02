package net.hudup.regression.em;

import static net.hudup.regression.AbstractRM.notSatisfy;
import static net.hudup.regression.AbstractRM.splitIndices;
import static net.hudup.regression.em.REMImpl.R_CALC_VARIANCE_FIELD;

import java.awt.Color;
import java.util.Arrays;
import java.util.List;

import flanagan.analysis.Regression;
import flanagan.math.Fmath;
import flanagan.plot.PlotGraph;
import net.hudup.core.Constants;
import net.hudup.core.Util;
import net.hudup.core.alg.Alg;
import net.hudup.core.alg.DuplicatableAlg;
import net.hudup.core.data.AttributeList;
import net.hudup.core.data.DataConfig;
import net.hudup.core.data.Fetcher;
import net.hudup.core.data.Profile;
import net.hudup.core.logistic.MathUtil;
import net.hudup.core.logistic.Vector2;
import net.hudup.regression.LargeStatistics;
import net.hudup.regression.VarWrapper;
import net.hudup.regression.em.ui.graph.Graph;
import net.hudup.regression.em.ui.graph.PlotGraphExt;

/**
 * This class implements the semi-mixture regression model.
 * If there are 2 partial models and dual mode is true, this semi-mixture regression model becomes dual regression model.
 *  
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class SemiMixtureREM extends AbstractMixtureREM implements DuplicatableAlg {

	
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

	
	@Override
	protected boolean prepareInternalData(Fetcher<Profile> inputSample) throws Exception {
		clearInternalData();
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
			REMImpl rem = createREM();
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
	protected REMImpl createREM() {
		// TODO Auto-generated method stub
		REMImpl rem = super.createREM();
		rem.getConfig().put(EM_EPSILON_FIELD, this.getConfig().get(EM_EPSILON_FIELD));
		rem.getConfig().put(EM_MAX_ITERATION_FIELD, this.getConfig().get(EM_MAX_ITERATION_FIELD));
		rem.getConfig().put(R_CALC_VARIANCE_FIELD, false);
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
			LargeStatistics stat0 = this.getLargeStatistics();
			int N = stat0.getZData().size();
			for (REMImpl rem : this.rems) {
				LargeStatistics stat = rem.getLargeStatistics();
				if (stat.getZData().size() != N)
					return null;
			}
			
			//Calculating average value of Z in mutual mode.
			for (int i = 0; i < N; i++) {
				double mean0 = 0;
				double coeffSum = 0;
				for (REMImpl rem : this.rems) {
					ExchangedParameter parameter = rem.getExchangedParameter();
					LargeStatistics stat = rem.getLargeStatistics();
					double mean = parameter.mean(stat.getXData().get(i));
					double coeff = parameter.getCoeff();
					coeff = Util.isUsed(coeff) ? coeff : 1;
					mean0 += coeff * mean;
					coeffSum += coeff;
				}
				mean0 = mean0 / coeffSum;
				
				for (REMImpl rem : this.rems) {
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
		
		for (REMImpl rem : this.rems) {
			ExchangedParameter parameter = rem.getExchangedParameter();
			double zVariance = parameter.estimateZVariance(rem.getLargeStatistics());
			parameter.setZVariance(zVariance);
			parameter.setCoeff(1.0 / (double)this.rems.size());
		}
		//In uniform mode, all coefficients are 1/K. In logistic mode, coefficients are not used. 
		if (getConfig().getAsBoolean(UNIFORM_MODE_FIELD))
			return true;
		
		List<ExchangedParameter> parameterList = Util.newList(this.rems.size());
		for (REMImpl rem : this.rems) {
			ExchangedParameter parameter = rem.getExchangedParameter();
			parameterList.add((ExchangedParameter)parameter.clone());
		}
		
		this.currentIteration++;
		for (int k = 0; k < this.rems.size(); k++) {
			REMImpl rem = this.rems.get(k);
			ExchangedParameter parameter = rem.getExchangedParameter();
			
			double condProbSum = 0;
			int N = 0;
			List<double[]> zData = rem.getData().getZData(); //By default, all models have the same original Z variables.
			for (int i = 0; i < zData.size(); i++) {
				double zValue = zData.get(i)[1];
				if (!Util.isUsed(zValue))
					continue;
				
				List<double[]> XList = Util.newList(this.rems.size());
				for (REMImpl rem2 : this.rems) {
					XList.add(rem2.getLargeStatistics().getXData().get(i));
				}
				
				List<Double> condProbs = ExchangedParameter.normalZCondProbs(parameterList, XList, Arrays.asList(new double[] {1, zValue}));
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
		
		for (REMImpl rem : this.rems) {
			ExchangedParameter parameter = rem.getExchangedParameter();
			double zVariance = parameter.estimateZVariance(rem.getLargeStatistics());
			parameter.setZVariance(zVariance);
			parameter.setCoeff(1.0 / (double)this.rems.size());
		}
		//In uniform mode, all coefficients are 1/K. In logistic mode, coefficients are not used. 
		if (getConfig().getAsBoolean(UNIFORM_MODE_FIELD))
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
			for (REMImpl rem : this.rems) {
				ExchangedParameter parameter = rem.getExchangedParameter();
				parameterList.add((ExchangedParameter)parameter.clone());
			}
			
			for (int k = 0; k < this.rems.size(); k++) {
				REMImpl rem = this.rems.get(k);
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
					for (REMImpl rem2 : this.rems) {
						XList.add(rem2.getLargeStatistics().getXData().get(i));
					}
					
					List<Double> condProbs = ExchangedParameter.normalZCondProbs(parameterList, XList, Arrays.asList(new double[] {1, zValue}));
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
	public synchronized LargeStatistics getLargeStatistics() {
		// TODO Auto-generated method stub
		if (this.rems == null || this.rems.size() == 0)
			return null;
		
		int N  = this.rems.get(0).getLargeStatistics().size();
		List<double[]> xData = Util.newList(N);
		List<double[]> zData = Util.newList(N);
		for (int i = 0; i < N; i++) {
			int K = this.rems.size();
			double[] xVector = new double[K + 1];
			xVector[0] = 1;
			double[] zVector = new double[2];
			zVector[0] = 1;
			
			double zValue = 0;
			for (int k = 0; k < K; k++) {
				LargeStatistics stats = this.rems.get(k).getLargeStatistics();
				xVector[k + 1] = stats.getXData().get(i)[1];
				
				double coeff = this.rems.get(k).getExchangedParameter().getCoeff();
				zValue += coeff * stats.getZData().get(i)[1];
			}
			zVector[1] = zValue;
			
			xData.add(xVector);
			zData.add(zVector);
		}
		
		return new LargeStatistics(xData, zData);
	}


	@Override
	public synchronized double executeByXStatistic(double[] xStatistic) {
		// TODO Auto-generated method stub
		if (this.rems == null || this.rems.size() == 0 || xStatistic.length != this.rems.size() + 1)
			return Constants.UNUSED;
		
		double result = 0;
		for (int k = 0; k < this.rems.size(); k++) {
			REMImpl rem = this.rems.get(k);
			double value = rem.executeByXStatistic(new double[] {1, xStatistic[k + 1]});
			if (Util.isUsed(value))
				result += rem.getExchangedParameter().getCoeff() * value;
			else
				return Constants.UNUSED;
		}
		
		return result;
	}


	@Override
	public synchronized Object execute(Object input) {
		// TODO Auto-generated method stub
//		if (getConfig().getAsBoolean(LOGISTIC_MODE_FIELD)) { // Logistic mode does not use probability
//			if (this.rems == null || this.rems.size() == 0)
//				return null;
//			
//			List<Double> zValues = Util.newList(this.rems.size());
//			List<Double> expProbs = Util.newList(this.rems.size());
//			double expProbsSum = 0;
//			for (int k = 0; k < this.rems.size(); k++) {
//				REMImpl rem = this.rems.get(k);
//				double zValue = extractNumber(rem.execute(input));
//				if (!Util.isUsed(zValue))
//					return null;
//				
//				zValues.add(zValue);
//				
//				ExchangedParameter parameter = rem.getExchangedParameter();
//				double prob = ExchangedParameter.normalPDF(zValue, 
//						parameter.mean(rem.extractRegressorValues(input)),
//						parameter.getZVariance());
//				double weight = Math.exp(prob);
//				expProbs.add(weight);
//				expProbsSum += weight;
//			}
//
//			double result = 0;
//			for (int k = 0; k < this.rems.size(); k++) {
//				result += (expProbs.get(k) / expProbsSum) * zValues.get(k); 
//			}
//			
//			return result;
//		}
//		else
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
		SemiMixtureREM semiMixREM = new SemiMixtureREM();
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
		config.remove(COMP_EXECUTION_FIELD);
		
		config.addReadOnly(DUPLICATED_ALG_NAME_FIELD);
		return config;
	}

	
	@Override
	public VarWrapper extractRegressor(int index) {
		// TODO Auto-generated method stub
		if (this.rems == null || this.rems.size() == 0)
			return null;
		else
			return this.rems.get(index - 1).extractRegressor(1);
	}


	@Override
	public List<VarWrapper> extractRegressors() {
		// TODO Auto-generated method stub
		List<VarWrapper> varList = Util.newList();
		if (this.rems == null || this.rems.size() == 0)
			return varList;
		
		for (REMImpl rem : this.rems) {
			VarWrapper var = rem.extractRegressor(1);
			varList.add(var);
		}
		
		return varList;
	}


	@Override
	public List<VarWrapper> extractSingleRegressors() {
		// TODO Auto-generated method stub
		List<VarWrapper> varList = Util.newList();
		if (this.rems == null || this.rems.size() == 0)
			return varList;
		
		for (REMImpl rem : this.rems) {
			VarWrapper var = rem.extractSingleRegressors().get(0);
			varList.add(var);
		}
		
		return varList;
	}


	@Override
	public double extractRegressorValue(Object input, int index) {
		// TODO Auto-generated method stub
		if (this.rems == null || this.rems.size() == 0)
			return Constants.UNUSED;
		else
			return this.rems.get(index - 1).extractRegressorValue(input, 1);
	}


	@Override
	public List<Double> extractRegressorStatistic(VarWrapper regressor) {
		// TODO Auto-generated method stub
		if (this.rems == null || this.rems.size() == 0)
			return Util.newList();
		
		for (REMImpl rem : this.rems) {
			List<VarWrapper> varList = rem.extractRegressors();
			for (VarWrapper var : varList) {
				if (var.equals(regressor))
					return rem.extractRegressorStatistic(var);
			}
		}
		
		return Util.newList();
	}


	@Override
	public synchronized Graph createRegressorGraph(VarWrapper regressor) {
		// TODO Auto-generated method stub
		if (this.rems == null || this.rems.size() == 0)
			return null;
		
		for (REMImpl rem : this.rems) {
			List<VarWrapper> varList = rem.extractRegressors();
			for (VarWrapper var : varList) {
				if (var.equals(regressor))
					return rem.createRegressorGraph(var);
			}
		}
		
		return null;
	}


	@Override
	public synchronized Graph createResponseGraph() {
		// TODO Auto-generated method stub
		if (this.rems == null || this.rems.size() == 0)
			return null;

		int ncurves = 2;
    	int npoints = this.getLargeStatistics().size();
    	double[][] data = PlotGraph.data(ncurves, npoints);

    	for(int i = 0; i < npoints; i++) {
			double z = 0;
			double zEstimated = 0;
			for (REMImpl rem : this.rems) {
				double coeff = rem.getExchangedParameter().getCoeff();
				double[] xVector = rem.getLargeStatistics().getXData().get(i);
				
				z += coeff * rem.getLargeStatistics().getZData().get(i)[1];
				zEstimated += coeff * rem.executeByXStatisticWithoutTransform(xVector);
			}
			z = (double)transformResponse(z, true);
			zEstimated = (double)transformResponse(zEstimated, true);
			
			data[0][i] = z;
            data[1][i] = zEstimated;
        }

    	Regression regression = new Regression(data[0], data[1]);
    	regression.linear();
    	double[] coef = regression.getCoeff();
    	data[2][0] = Fmath.minimum(data[0]);
    	data[3][0] = coef[0] + coef[1] * data[2][0];
    	data[2][1] = Fmath.maximum(data[0]);
    	data[3][1] = coef[0] + coef[1] * data[2][1];

    	PlotGraphExt pg = new PlotGraphExt(data) {

			/**
			 * Serial version UID for serializable class.
			 */
			private static final long serialVersionUID = 1L;

			@Override
			public String getGraphFeature() {
				// TODO Auto-generated method stub
				return "R=" + MathUtil.format(calcR(), 2);
			}
    		
    	};

    	pg.setGraphTitle("Correlation plot: " + pg.getGraphFeature());
    	pg.setXaxisLegend("Real " + transformResponse(extractResponse().toString(), true));
    	pg.setYaxisLegend("Estimated " + transformResponse(extractResponse().toString(), true));
    	int[] popt = {1, 0};
    	pg.setPoint(popt);
    	int[] lopt = {0, 3};
    	pg.setLine(lopt);

    	pg.setBackground(Color.WHITE);
        return pg;
	}


	@Override
	public Graph createErrorGraph() {
		// TODO Auto-generated method stub
		if (this.rems == null || this.rems.size() == 0)
			return null;
		
    	int ncurves = 4;
    	int npoints = this.getLargeStatistics().size();
    	double[][] data = PlotGraph.data(ncurves, npoints);

		double errorMean = 0;
    	for(int i = 0; i < npoints; i++) {
			double z = 0;
			double zEstimated = 0;
			for (REMImpl rem : this.rems) {
				double coeff = rem.getExchangedParameter().getCoeff();
				double[] xVector = rem.getLargeStatistics().getXData().get(i);
				
				z += coeff * rem.getLargeStatistics().getZData().get(i)[1];
				zEstimated += coeff * rem.executeByXStatisticWithoutTransform(xVector);
			}
			z = (double)transformResponse(z, true);
			zEstimated = (double)transformResponse(zEstimated, true);

			data[0][i] = ( z + zEstimated ) / 2.0;
            data[1][i] = zEstimated - z;
            errorMean += data[1][i];
        }
    	errorMean = errorMean / npoints;
    	double errorSd = 0;
    	for(int i = 0; i < npoints; i++) {
    		double d = data[1][i] - errorMean;
    		errorSd += d*d;
    	}
   		errorSd = Math.sqrt(errorSd / npoints); //MLE estimation
    		
    	// Mean - 1.96sd
    	data[2][0] = 0;
    	data[3][0] = errorMean - 1.96 * errorSd;
    	data[2][1] = Fmath.maximum(data[0]);
    	data[3][1] = errorMean - 1.96 * errorSd;

    	// Mean
    	data[4][0] = 0;
    	data[5][0] = errorMean;
    	data[4][1] = Fmath.maximum(data[0]);
    	data[5][1] = errorMean;

    	// Mean + 1.96sd
    	data[6][0] = 0;
    	data[7][0] = errorMean + 1.96 * errorSd;
    	data[6][1] = Fmath.maximum(data[0]);
    	data[7][1] = errorMean + 1.96 * errorSd;

    	final double mean = errorMean, sd = errorSd;
    	PlotGraphExt pg = new PlotGraphExt(data) {

			/**
			 * Serial version UID for serializable class.
			 */
			private static final long serialVersionUID = 1L;

			@Override
			public String getGraphFeature() {
				// TODO Auto-generated method stub
				return MathUtil.format(mean, 2) + " +/- 1.96*" + 
    				MathUtil.format(sd, 2);
			}
    		
    	};

    	pg.setGraphTitle("Error plot: " + pg.getGraphFeature());
    	pg.setXaxisLegend("Mean " + transformResponse(extractResponse().toString(), true));
    	pg.setYaxisLegend("Estimated error");
    	int[] popt = {1, 0, 0, 0};
    	pg.setPoint(popt);
    	int[] lopt = {0, 3, 3, 3};
    	pg.setLine(lopt);

    	pg.setBackground(Color.WHITE);
    	
        return pg;
	}


	@Override
	public synchronized double calcVariance() {
		// TODO Auto-generated method stub
		if (this.rems == null || this.rems.size() == 0)
			return Constants.UNUSED;
		
		int N = this.getLargeStatistics().size();
		double ss = 0;
		for (int i = 0; i < N; i++) {
			double z = 0;
			double zEstimated = 0;
			for (REMImpl rem : this.rems) {
				double coeff = rem.getExchangedParameter().getCoeff();
				double[] xVector = rem.getLargeStatistics().getXData().get(i);
				
				z += coeff * rem.getLargeStatistics().getZData().get(i)[1];
				zEstimated += coeff * rem.executeByXStatisticWithoutTransform(xVector);
			}
			z = (double)transformResponse(z, true);
			zEstimated = (double)transformResponse(zEstimated, true);
			
			ss += (zEstimated - z) * (zEstimated - z);
		}
		
		return ss / N;
	}


	@Override
	public synchronized double calcR() {
		// TODO Auto-generated method stub
		if (this.rems == null || this.rems.size() == 0)
			return Constants.UNUSED;
		
		int N = this.getLargeStatistics().size();
		Vector2 zVector = new Vector2(N, 0);
		Vector2 zEstimatedVector = new Vector2(N, 0);
		for (int i = 0; i < N; i++) {
			double z = 0;
			double zEstimated = 0;
			for (REMImpl rem : this.rems) {
				double coeff = rem.getExchangedParameter().getCoeff();
				double[] xVector = rem.getLargeStatistics().getXData().get(i);
				
				z += coeff * rem.getLargeStatistics().getZData().get(i)[1];
				zEstimated += coeff * rem.executeByXStatisticWithoutTransform(xVector);
			}
			z = (double)transformResponse(z, true);
			zEstimated = (double)transformResponse(zEstimated, true);
			
            zVector.set(i, z);
            zEstimatedVector.set(i, zEstimated);
		}
		
		return zEstimatedVector.corr(zVector);
	}


	@Override
	public double[] calcError() {
		// TODO Auto-generated method stub
		if (this.rems == null || this.rems.size() == 0)
			return null;
		
		int N = this.getLargeStatistics().size();
		Vector2 error = new Vector2(N, 0);
		for (int i = 0; i < N; i++) {
			double z = 0;
			double zEstimated = 0;
			for (REMImpl rem : this.rems) {
				double coeff = rem.getExchangedParameter().getCoeff();
				double[] xVector = rem.getLargeStatistics().getXData().get(i);
				
				z += coeff * rem.getLargeStatistics().getZData().get(i)[1];
				zEstimated += coeff * rem.executeByXStatisticWithoutTransform(xVector);
			}
			
            error.set(i, zEstimated - z);
		}
		
    	return new double[] {error.mean(), error.mleVar()};
	}


}
