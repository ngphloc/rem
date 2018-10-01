package net.hudup.regression.em;

import static net.hudup.regression.AbstractRM.createProfile;
import static net.hudup.regression.AbstractRM.extractNumber;
import static net.hudup.regression.AbstractRM.extractSingleVariables;
import static net.hudup.regression.AbstractRM.extractVariable;
import static net.hudup.regression.AbstractRM.extractVariableValue;
import static net.hudup.regression.AbstractRM.extractVariables;
import static net.hudup.regression.AbstractRM.findIndex;
import static net.hudup.regression.AbstractRM.notSatisfy;
import static net.hudup.regression.AbstractRM.parseIndices;
import static net.hudup.regression.AbstractRM.solve;

import java.awt.Color;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import javax.swing.JOptionPane;

import flanagan.math.Fmath;
import flanagan.plot.PlotGraph;
import net.hudup.core.Constants;
import net.hudup.core.Util;
import net.hudup.core.alg.Alg;
import net.hudup.core.alg.DuplicatableAlg;
import net.hudup.core.data.AttributeList;
import net.hudup.core.data.DataConfig;
import net.hudup.core.data.Fetcher;
import net.hudup.core.data.MemFetcher;
import net.hudup.core.data.Profile;
import net.hudup.core.logistic.DSUtil;
import net.hudup.core.logistic.MathUtil;
import net.hudup.core.logistic.ui.UIUtil;
import net.hudup.em.ExponentialEM;
import net.hudup.regression.AbstractRM;
import net.hudup.regression.LargeStatistics;
import net.hudup.regression.Statistics;
import net.hudup.regression.VarWrapper;
import net.hudup.regression.em.ui.REMDlg;
import net.hudup.regression.em.ui.graph.Graph;
import net.hudup.regression.em.ui.graph.PlotGraphExt;

/**
 * This class implements default expectation maximization algorithm for regression model in case of missing data, called REM algorithm. 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class REMImpl extends ExponentialEM implements REM, DuplicatableAlg {

	
	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Name of variance calculation field.
	 */
	public final static String R_CALC_VARIANCE_FIELD = "r_calc_variance";

	
	/**
	 * Default value variance calculation field.
	 */
	public final static boolean R_CALC_VARIANCE_DEFAULT = false;

	
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
	
	
	/**
	 * Default constructor.
	 */
	public REMImpl() {
		// TODO Auto-generated constructor stub
		super();
	}
	
	
	@Override
	public Object learn(Object...info) throws Exception {
		// TODO Auto-generated method stub
		Object resulted = null;
		if (prepareInternalData(this.sample))
			resulted = super.learn();
		if (resulted == null)
			clearInternalData();

		return resulted;
	}


	/**
	 * Preparing data.
	 * @param inputSample specified sample.
	 * @return true if data preparation is successful.
	 * @throws Exception if any error raises.
	 */
	protected boolean prepareInternalData(Fetcher<Profile> inputSample) throws Exception {
		clearInternalData();
		
		this.attList = getSampleAttributeList(inputSample);
		if (this.attList.size() < 2)
			return false;

		//Begin parsing indices
		String cfgIndices = this.getConfig().getAsString(R_INDICES_FIELD);
		if (!parseIndices(cfgIndices, this.attList.size(), this.xIndices, this.zIndices)) //parsing indices
			return false;
		//End parsing indices
		
		//Begin checking existence of values.
		boolean zExists = false;
		boolean[] xExists = new boolean[this.xIndices.size() - 1]; //profile = (x1, x2,..., x(n-1), z)
		Arrays.fill(xExists, false);
		while (inputSample.next()) {
			Profile profile = inputSample.pick(); //profile = (x1, x2,..., x(n-1), z)
			if (profile == null)
				continue;
			
			double lastValue = extractNumber(extractResponseValue(profile));
			if (Util.isUsed(lastValue))
				zExists = zExists || true; 
			
			for (int j = 1; j < this.xIndices.size(); j++) {
				double value = extractRegressorValue(profile, j);
				if (Util.isUsed(value))
					xExists[j - 1] = xExists[j - 1] || true;
			}
		}
		inputSample.reset();
		List<Object[]> xIndicesTemp = Util.newList();
		xIndicesTemp.add(this.xIndices.get(0)); //adding -1
		for (int j = 1; j < this.xIndices.size(); j++) {
			if (xExists[j - 1])
				xIndicesTemp.add(this.xIndices.get(j)); //only use variables having at least one value.
		}
		if (!zExists || xIndicesTemp.size() < 2)
			return false;
		this.xIndices = xIndicesTemp;
		//End checking existence of values.
		
		//Begin extracting data
		List<double[]> xData = Util.newList();
		List<double[]> zData = Util.newList();
		while (inputSample.next()) {
			Profile profile = inputSample.pick(); //profile = (x1, x2,..., x(n-1), z)
			if (profile == null)
				continue;
			
			double[] xVector = new double[this.xIndices.size()]; //1, x1, x2,..., x(n-1)
			double[] zVector = new double[2]; //1, z
			xVector[0] = 1.0;
			zVector[0] = 1.0;
			
			double lastValue = extractNumber(extractResponseValue(profile));
			if (!Util.isUsed(lastValue))
				zVector[1] = Constants.UNUSED;
			else
				zVector[1] = (double)transformResponse(lastValue, false);
			
			for (int j = 1; j < this.xIndices.size(); j++) {
				double value = extractRegressorValue(profile, j);
				if (!Util.isUsed(value))
					xVector[j] = Constants.UNUSED;
				else
					xVector[j] = (double)transformRegressor(value, false);
			}
			
			zData.add(zVector);
			xData.add(xVector);
		}
		inputSample.reset();
		//End extracting data
		
		if (xData.size() == 0 || zData.size() == 0)
			return false;
		else {
			this.data = new LargeStatistics(xData, zData);
			return true;
		}
	}
	
	
	/**
	 * Setting internal data.
	 * @param xIndices specified X indices.
	 * @param zIndices specified Z indices.
	 * @param attList specified attribute list.
	 * @param data specified data.
	 * @return true if setting successful.
	 */
	protected boolean prepareInternalData(List<Object[]> xIndices, List<Object[]> zIndices, AttributeList attList, LargeStatistics data) {
		clearInternalData();
		this.xIndices = xIndices;
		this.zIndices = zIndices;
		this.attList = attList;
		this.data = data;
		return true;
	}
	
	
	/**
	 * Clear all internal data.
	 */
	protected void clearInternalData() {
		this.currentIteration = 0;
		this.currentParameter = this.estimatedParameter = null;
		this.xIndices.clear();
		this.zIndices.clear();
		this.attList = null;
		
		if (this.statistics != null && (this.statistics instanceof LargeStatistics))
			((LargeStatistics)this.statistics).clear();
		this.statistics = null;
		
		if (this.data != null)
			this.data.clear();
		this.data = null;
	}
	
	
	/**
	 * Expectation method of this class does not change internal data.
	 */
	@Override
	protected Object expectation(Object currentParameter, Object...info) throws Exception {
		// TODO Auto-generated method stub
		if (currentParameter == null)
			return null;
		List<Double> alpha = ((ExchangedParameter)currentParameter).getAlpha();
		List<double[]> betas = ((ExchangedParameter)currentParameter).getBetas();
		LargeStatistics data = null;
		if (info != null && info.length > 0 && (info[0] instanceof LargeStatistics))
			data = (LargeStatistics)info[0];
		else
			data = this.data;
		
		int N = data.getZData().size();
		List<double[]> zStatistic = Util.newList(N);
		List<double[]> xStatistic = Util.newList(N);
		for (int i = 0; i < N; i++) {
			Statistics stat0 = new Statistics(data.getZData().get(i)[1], data.getXData().get(i));
			Statistics stat = estimate(stat0, alpha, betas);
			if (stat == null)
				return null;
			
			stat = (stat.checkValid() ? stat : null);
			if (stat == null)
				return null;
			zStatistic.add(new double[] {1.0, stat.getZStatistic()});
			xStatistic.add(stat.getXStatistic());
		}
		
		return new LargeStatistics(xStatistic, zStatistic);
	}

	
	/**
	 * Expectation method of this class does not change internal data.
	 */
	@Override
	protected Object maximization(Object currentStatistic, Object...info) throws Exception {
		// TODO Auto-generated method stub
		LargeStatistics stat = (LargeStatistics)currentStatistic;
		if (stat == null || stat.isEmpty())
			return null;
		List<double[]> xStatistic = stat.getXData();
		List<double[]> zStatistic = stat.getZData();
		int N = zStatistic.size();
		int n = xStatistic.get(0).length; //1, x1, x2,..., x(n-1)
		ExchangedParameter currentParameter = (ExchangedParameter)getCurrentParameter();
		
		List<Double> alpha = calcCoeffsByStatistics(xStatistic, zStatistic);
		if (alpha == null || alpha.size() == 0) { //If cannot calculate alpha by matrix calculation.
			if (currentParameter != null)
				alpha = DSUtil.toDoubleList(currentParameter.getAlpha()); //clone alpha
			else { //Used for initialization so that regression model is always determined.
				alpha = DSUtil.initDoubleList(n, 0.0);
				double alpha0 = 0;
				for (int i = 0; i < N; i++)
					alpha0 += zStatistic.get(i)[1];
				alpha.set(0, alpha0 / (double)N); //constant function z = c
			}
		}
		
		List<double[]> betas = Util.newList(n);
		for (int j = 0; j < n; j++) {
			if (j == 0) {
				double[] beta0 = new double[2];
				beta0[0] = 1;
				beta0[1] = 0;
				betas.add(beta0);
				continue;
			}
			
			List<double[]> Z = Util.newList(N);
			List<Double> x = Util.newList(N);
			for (int i = 0; i < N; i++) {
				Z.add(zStatistic.get(i));
				x.add(xStatistic.get(i)[j]);
			}
			List<Double> beta = calcCoeffs(Z, x);
			if (beta == null || beta.size() == 0) {
				if (currentParameter != null)
					beta = DSUtil.toDoubleList(currentParameter.getBetas().get(j));
				else { //Used for initialization so that regression model is always determined.
					beta = DSUtil.initDoubleList(2, 0);
					double beta0 = 0;
					for (int i = 0; i < N; i++)
						beta0 += xStatistic.get(i)[j];
					beta.set(0, beta0 / (double)N); //constant function x = c
				}
			}
			betas.add(DSUtil.toDoubleArray(beta));
		}
		
		ExchangedParameter newParameter = new ExchangedParameter(alpha, betas);
		if (currentParameter != null)
			newParameter.setCoeff(currentParameter.getCoeff());
		if (getConfig().getAsBoolean(R_CALC_VARIANCE_FIELD))
			newParameter.setZVariance(newParameter.estimateZVariance(stat));
		else {
			if (currentParameter == null)
				newParameter.setZVariance(Constants.UNUSED);
			else if (Util.isUsed(currentParameter.getZVariance()))
				newParameter.setZVariance(newParameter.estimateZVariance(stat));
			else
				newParameter.setZVariance(Constants.UNUSED);
		}
		
		return newParameter;
	}
	
	
	/**
	 * Estimating statistics with specified parameters alpha and beta. This method does not change internal data.
	 * Balance process is removed because it is over-fitting or not stable. Balance process is the best in some cases.
	 * @param stat specified statistics.
	 * @param alpha specified alpha parameter.
	 * @param betas specified alpha parameters.
	 * @return estimated statistics with specified parameters alpha and beta. Return null if any error raises.
	 */
	protected Statistics estimate(Statistics stat, List<Double> alpha, List<double[]> betas) {
		double zValue = stat.getZStatistic();
		double[] xVector = stat.getXStatistic();
		double zStatistic = Constants.UNUSED;
		double[] xStatistic = new double[xVector.length];
		
		if (Util.isUsed(zValue)) {
			zStatistic = zValue;
			
			//Estimating missing xij (xStatistic) by equation 5 and zi (zStatistic) above, based on current parameter.
			for (int j = 0; j < xVector.length; j++) {
				if (Util.isUsed(xVector[j]))
					xStatistic[j] = xVector[j];
				else
					xStatistic[j] = betas.get(j)[0] + betas.get(j)[1] * zStatistic;
			}
			
			return new Statistics(zStatistic, xStatistic);
		}
		
		//Estimating missing zi (zStatistic) by equation 7, based on current parameter.
		double a = 0, b = 0, c = 0;
		List<Integer> U = Util.newList(); //Indices of missing values.
		for (int j = 0; j < xVector.length; j++) {
			if (Util.isUsed(xVector[j])) {
				b += alpha.get(j) * xVector[j];
			}
			else {
				a += alpha.get(j) * betas.get(j)[0];
				c += alpha.get(j) * betas.get(j)[1];
				U.add(j);
			}
		}
		if (c != 1) {
			zStatistic = (a + b) / (1.0 - c);
		}
		else {
			logger.info("Cannot estimate statistic for Z by expectation (#estimate), stop estimating for this statistic here because use of other method is wrong.");
			return null;
		}
		
		//Estimating missing xij (xStatistic) by equation 5 and estimated zi (zStatistic) above, based on current parameter.
		for (int j = 0; j < xVector.length; j++) {
			if (Util.isUsed(xVector[j]))
				xStatistic[j] = xVector[j];
			else
				xStatistic[j] = betas.get(j)[0] + betas.get(j)[1] * zStatistic;
		}
		
		//Balance process is removed because it is not necessary. Balance process is the best in some cases. So list U is not used.
		return new Statistics(zStatistic, xStatistic);
	}

	
	/**
	 * Estimating statistics with specified parameters alpha and beta. This method does not change internal data.
	 * Balance process is removed because it is over-fitting or not stable. Balance process is the best in some cases.
	 * This method is as good as than {@link #estimate(Statistics, List, List, Statistics)} method but it is not stable for long regression model having many regressors
	 * because solving a set of many equations can cause approximate solution or non-solution problem.   
	 * @param stat specified statistics.
	 * @param alpha specified alpha parameter.
	 * @param betas specified alpha parameters.
	 * @return estimated statistics with specified parameters alpha and beta. Return null if any error raises.
	 */
	@Deprecated
	protected Statistics estimateInverse(Statistics stat, List<Double> alpha, List<double[]> betas) {
		double zValue = stat.getZStatistic();
		double[] xVector = stat.getXStatistic();
		double zStatistic = Constants.UNUSED;
		double[] xStatistic = new double[xVector.length];
		
		if (Util.isUsed(zValue)) {
			zStatistic = zValue;
			//Estimating missing xij (xStatistic) by equation 5 and zi (zStatistic) above, based on current parameter.
			for (int j = 0; j < xVector.length; j++) {
				if (Util.isUsed(xVector[j]))
					xStatistic[j] = xVector[j];
				else
					xStatistic[j] = betas.get(j)[0] + betas.get(j)[1] * zStatistic;
			}
			
			return new Statistics(zStatistic, xStatistic);
		}
		
		List<Integer> U = Util.newList();
		double b = 0;
		for (int j = 0; j < xVector.length; j++) {
			if (Util.isUsed(xVector[j])) {
				b += alpha.get(j) * xVector[j];
				xStatistic[j] = xVector[j]; //existent xij
			}
			else
				U.add(j);
		}

		if (U.size() > 0) {
			//Estimating missing xij (xStatistic) by equation 8, based on current parameter.
			List<double[]> A = Util.newList(U.size());
			List<Double> y = Util.newList(U.size());
			
			for (int i = 0; i < U.size(); i++) {
				double[] aRow = new double[U.size()];
				A.add(aRow);
				for (int j = 0; j < U.size(); j++) {
					if (i == j)
						aRow[j] = betas.get(U.get(i))[1] * alpha.get(U.get(j)) - 1;
					else
						aRow[j] = betas.get(U.get(i))[1] * alpha.get(U.get(j));
				}
				double yi = -betas.get(U.get(i))[0] - betas.get(U.get(i))[1] * b;
				y.add(yi);
			}
			
			List<Double> solution = solve(A, y); //solve Ax = y
			if (solution != null) {
				for (int j = 0; j < U.size(); j++) {
					int k = U.get(j);
					xStatistic[k] = solution.get(j);
				}
			}
			else {
				logger.info("Cannot estimate statistic for X by expectation (#estimateInverse), stop estimating for this statistic here because use of other method is wrong.");
				return null;
			}
		}
		
		//Estimating missing zi (zStatistic) by equation 4, based on current parameter.
		zStatistic = 0;
		for (int j = 0; j < xStatistic.length; j++) {
			zStatistic += alpha.get(j) * xStatistic[j];
		}
		
		//Balance process is removed because it is not necessary. Balance process is the best in some cases. So list U is not used.
		return new Statistics(zStatistic, xStatistic);
	}

	
	/**
	 * Balancing missing values zi (xStatistic) and xij (xValues). This method does not change internal data.
	 * @param alpha alpha coefficients.
	 * @param betas beta coefficients.
	 * @param zStatistic statistic for Z variable.
	 * @param xStatistic statistic for X variables.
	 * @param U list of missing X values.
	 * @param inverse if true, this is inverse mode.
	 * @return balanced statistics for Z and X variables. Return null if any error raises.
	 */
	@Deprecated
	protected Statistics balanceStatistics(List<Double> alpha, List<double[]> betas,
			double zStatistic, double[] xStatistic,
			List<Integer> U, boolean inverse) {

		double zStatisticNext = Constants.UNUSED;
		double[] xStatisticNext = new double[xStatistic.length];
		int t = 0;
		int maxIteration = getConfig().getAsInt(EM_MAX_ITERATION_FIELD);
		maxIteration = (maxIteration <= 0) ? EM_MAX_ITERATION : maxIteration;
		double threshold = getConfig().getAsReal(EM_EPSILON_FIELD);
		while (t < maxIteration) {
			if (!inverse) {
				zStatisticNext = 0;
				for (int j = 0; j < xStatistic.length; j++)
					zStatisticNext += alpha.get(j) * xStatistic[j];
				
				for (int j = 0; j < xStatistic.length; j++) {
					if (!U.contains(j))
						xStatisticNext[j] = xStatistic[j];
					else
						xStatisticNext[j] = betas.get(j)[0] + betas.get(j)[1] * zStatisticNext;
				}
				
			}
			else {
				for (int j = 0; j < xStatistic.length; j++) {
					if (!U.contains(j))
						xStatisticNext[j] = xStatistic[j];
					else
						xStatisticNext[j] = betas.get(j)[0] + betas.get(j)[1] * zStatistic;
				}
				
				zStatisticNext = 0;
				for (int j = 0; j < xStatistic.length; j++)
					zStatisticNext += alpha.get(j) * xStatisticNext[j];
			}
			
			t++;
			
			//Testing approximation
			boolean approx = !notSatisfy(zStatisticNext, zStatistic, threshold);
			for (int j = 0; j < xStatistic.length; j++) {
				approx = approx && !notSatisfy(xStatisticNext[j], xStatistic[j], threshold);
				if (!approx) break;
			}
			
			zStatistic = zStatisticNext;
			xStatistic = xStatisticNext;
			zStatisticNext = Constants.UNUSED;
			xStatisticNext = new double[xStatistic.length];
			
			if (approx) break;
		} //If the likelihood function is too acute, the loop can be infinite.
		
		return new Statistics(zStatistic, xStatistic);
	}

	
	/**
	 * Initialization method of this class does not change internal data.
	 */
	@Override
	protected Object initializeParameter() {
		// TODO Auto-generated method stub
		int n = this.data.getXData().get(0).length;
		ExchangedParameter parameter0 = initializeAlphaBetas(n, false);
		
		LargeStatistics completeData = getCompleteData(this.data);
		if (completeData == null)
			return parameter0;
			
		try {
			ExchangedParameter parameter = (ExchangedParameter) maximization(completeData);
			return (parameter != null ? parameter : parameter0); 
		}
		catch (Throwable e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		return parameter0;
	}

	
	@Override
	protected boolean terminatedCondition(Object estimatedParameter, Object currentParameter, Object previousParameter, Object... info) {
		// TODO Auto-generated method stub
		double threshold = getConfig().getAsReal(EM_EPSILON_FIELD);
		
		return ((ExchangedParameter)estimatedParameter).terminatedCondition(
				threshold, 
				(ExchangedParameter)currentParameter, 
				(ExchangedParameter)previousParameter);
	}

	
	/**
	 * Getting exchanged parameter. Actually, this method calls {@link #getParameter()}.
	 * @return exchanged parameter.
	 */
	public ExchangedParameter getExchangedParameter() {
		return (ExchangedParameter)getParameter();
	}
	
	
	/**
	 * Getting large statistics. Actually, this method calls {@link #getStatistics()}.
	 * @return large statistics.
	 */
	public LargeStatistics getLargeStatistics() {
		return (LargeStatistics)getStatistics();
	}
	
	
	@Override
	public synchronized Object executeByXStatistic(double[] xStatistic) {
		if (xStatistic == null)
			return null;

		ExchangedParameter parameter = this.getExchangedParameter(); 
		if (parameter == null)
			return null;
		List<Double> alpha = parameter.getAlpha();

		Statistics stat = estimate(new Statistics(Constants.UNUSED, xStatistic), alpha, parameter.getBetas());
		if (stat == null)
			return null;
		else
			return transformResponse(stat.getZStatistic(), true);
	}
	
	
	/**
	 * Executing by X statistics without transform.
	 * @param xStatistic X statistics (regressors). The first element of this X statistics is 1.
	 * @return result of execution without transform. Return null if execution is failed.
	 */
	public synchronized Object executeByXStatisticWithoutTransform(double[] xStatistic) {
		if (xStatistic == null)
			return null;

		ExchangedParameter parameter = this.getExchangedParameter(); 
		if (parameter == null)
			return null;
		List<Double> alpha = parameter.getAlpha();

		Statistics stat = estimate(new Statistics(Constants.UNUSED, xStatistic), alpha, parameter.getBetas());
		if (stat == null)
			return null;
		else
			return stat.getZStatistic();
	}

	
	/**
	 * This method can be used to estimate Z value with incomplete profile.
	 * In other words, it is possible to test with incomplete testing data.
	 */
	@Override
	public synchronized Object execute(Object input) {
		// TODO Auto-generated method stub
		double[] xStatistic = extractRegressorValues(input);
		return executeByXStatistic(xStatistic);
	}
	
	
	/**
	 * Executing this algorithm by arbitrary input parameter.
	 * @param input arbitrary input parameter.
	 * @return result of execution. Return null if execution is failed.
	 */
	public Object executeIntel(Object...input) {
		return execute(input);
	}
	
	
	/**
	 * Getting attribute list.
	 * @return attribute list.
	 */
	public AttributeList getAttributeList() {
		return this.attList;
	}
	
	
	/**
	 * Getting internal data. Actually, this method returns the current statistics.
	 * @return internal data which is the current statistics.
	 */
	protected synchronized LargeStatistics getData() {
		return this.data;
	}
	
	
	/**
	 * Testing whether missing values are fulfilled.
	 * @return true if missing values are fulfilled.
	 */
	public synchronized boolean isMissingDataFilled() {
		return (this.statistics != null && this.statistics != this.data);
	}
	
	
	@Override
	public String getName() {
		// TODO Auto-generated method stub
		String name = getConfig().getAsString(DUPLICATED_ALG_NAME_FIELD);
		if (name != null && !name.isEmpty())
			return name;
		else
			return "rem";
	}

	
	@Override
	public void setName(String name) {
		// TODO Auto-generated method stub
		getConfig().put(DUPLICATED_ALG_NAME_FIELD, name);
	}


	@Override
	public Alg newInstance() {
		// TODO Auto-generated method stub
		REMImpl em = new REMImpl();
		em.getConfig().putAll((DataConfig)this.getConfig().clone());
		return em;
	}

	
	@Override
	public DataConfig createDefaultConfig() {
		// TODO Auto-generated method stub
		DataConfig config = super.createDefaultConfig();
		config.put(R_INDICES_FIELD, R_INDICES_DEFAULT);
		config.put(R_CALC_VARIANCE_FIELD, R_CALC_VARIANCE_DEFAULT); //This attribute is used for testing
		config.addReadOnly(DUPLICATED_ALG_NAME_FIELD);
		return config;
	}

	
	@Override
	public synchronized String getDescription() {
		// TODO Auto-generated method stub
		if (this.getParameter() == null)
			return "";
		ExchangedParameter exParameter = ((ExchangedParameter)this.getParameter());
		List<Double> alpha = exParameter.getAlpha();
		if (alpha == null || alpha.size() == 0)
			return "";
		
		StringBuffer buffer = new StringBuffer();
		buffer.append(transformResponse(extractResponse().toString(), false) + " = " + MathUtil.format(alpha.get(0)));
		for (int j = 0; j < alpha.size() - 1; j++) {
			double coeff = alpha.get(j + 1);
			String regressorExpr = "(" + transformRegressor(extractRegressor(j + 1).toString(), false).toString() + ")";
			if (coeff < 0)
				buffer.append(" - " + MathUtil.format(Math.abs(coeff)) + "*" + regressorExpr);
			else
				buffer.append(" + " + MathUtil.format(coeff) + "*" + regressorExpr);
		}
		
		buffer.append(": ");
		buffer.append("t=" + getCurrentIteration());
		buffer.append(", coeff=" + MathUtil.format(exParameter.getCoeff()));
		buffer.append(", z-variance=" + MathUtil.format(exParameter.getZVariance()));

		return buffer.toString();
	}


	@Override
	public String parameterToShownText(Object parameter, Object...info) {
		// TODO Auto-generated method stub
		if (parameter == null || !(parameter instanceof ExchangedParameter))
			return "";
		
		ExchangedParameter exParameter = ((ExchangedParameter)parameter);
		return exParameter.toString();
	}

	
	@Override
	public synchronized void manifest() {
		// TODO Auto-generated method stub
		if (getParameter() == null) {
			JOptionPane.showMessageDialog(
					UIUtil.getFrameForComponent(null), 
					"Invalid regression model", 
					"Invalid regression model", 
					JOptionPane.ERROR_MESSAGE);
		}
		else
			new REMDlg(UIUtil.getFrameForComponent(null), this);
	}


	/**
	 * Calculating coefficients based on regressors X (statistic X) and response variable Z (statistic Z).
	 * Both statistic X and statistic Z contain 1 at first column.
	 * @param xStatistic regressors X (statistic X).
	 * @param zStatistic response variable Z (statistic Z).
	 * @return coefficients based on regressors X (statistic X) and response variable Z (statistic Z). Return null if any error raises.
	 */
	protected static List<Double> calcCoeffsByStatistics(List<double[]> xStatistic, List<double[]> zStatistic) {
		List<Double> z = Util.newList(zStatistic.size());
		for (int i = 0; i < zStatistic.size(); i++) {
			z.add(zStatistic.get(i)[1]);
		}
		
		return calcCoeffs(xStatistic, z);
	}
	
	
	/**
	 * Calculating coefficients based on data matrix and data vector.
	 * This method will be improved in the next version.
	 * @param X specified data matrix.
	 * @param z specified data vector.
	 * @return coefficients base on data matrix and data vector. Return null if any error raises.
	 */
	protected static List<Double> calcCoeffs(List<double[]> X, List<Double> z) {
		int N = z.size();
		int n = X.get(0).length;
		
		List<double[]> A = Util.newList(n);
		for (int i = 0; i < n; i++) {
			double[] aRow = new double[n];
			A.add(aRow);
			for (int j = 0; j < n; j++) {
				double sum = 0;
				for (int k = 0; k < N; k++) {
					sum += X.get(k)[i] * X.get(k)[j];
				}
				aRow[j] = sum;
			}
		}
		
		List<Double> b = Util.newList(n);
		for (int i = 0; i < n; i++) {
			double sum = 0;
			for (int k = 0; k < N; k++)
				sum += X.get(k)[i] * z.get(k);
			
			b.add(sum);
		}
		
		return solve(A, b);
	}
	
	
	@Override
	public VarWrapper extractRegressor(int index) {
		// TODO Auto-generated method stub
		return extractVariable(attList, xIndices, index);
	}

	
	@Override
	public List<VarWrapper> extractRegressors() {
		// TODO Auto-generated method stub
		return extractVariables(attList, xIndices);
	}


	@Override
	public List<VarWrapper> extractSingleRegressors() {
		// TODO Auto-generated method stub
		return extractSingleVariables(attList, xIndices);
	}


	@Override
	public double extractRegressorValue(Object input, int index) {
		// TODO Auto-generated method stub
		if (input == null)
			return Constants.UNUSED;
		else if (input instanceof Profile)
			return extractVariableValue(input, null, xIndices, index);
		else
			return extractVariableValue(input, attList, xIndices, index);
	}


	/**
	 * Extract regressors from input object.
	 * @param input specified input object.
	 * @return list of values of regressors from input object.
	 */
	protected double[] extractRegressorValues(Object input) {
		Profile profile = null;
		if (input instanceof Profile)
			profile = (Profile)input;
		else
			profile = createProfile(this.attList, input);
		if (profile == null)
			return null;
		
		double[] xStatistic = new double[this.xIndices.size()];
		xStatistic[0] = 1;
		for (int j = 1; j < this.xIndices.size(); j++) {
			double xValue = extractRegressorValue(profile, j);
			if (Util.isUsed(xValue))
				xStatistic[j] = (double)transformRegressor(xValue, false);
			else
				xStatistic[j] = Constants.UNUSED;
		}
		
		return xStatistic;
	}
	
	
	@Override
	public VarWrapper extractResponse() {
		// TODO Auto-generated method stub
		return extractVariable(attList, zIndices, 1);
	}


	@Override
	public synchronized Object extractResponseValue(Object input) {
		// TODO Auto-generated method stub
		if (input == null)
			return Constants.UNUSED;
		else if (input instanceof Profile)
			return extractVariableValue(input, null, zIndices, 1);
		else
			return extractVariableValue(input, attList, zIndices, 1);
	}


	/**
	 * Transforming independent variable X.
	 * In the most general case that each index is an mathematical expression, this method is not focused.
	 * @param x specified variable X.
	 * @param inverse if true, there is an inverse transformation.
	 * @return transformed value of X.
	 */
	protected Object transformRegressor(Object x, boolean inverse) {
		// TODO Auto-generated method stub
		return x;
	}


	@Override
	public Object transformResponse(Object z, boolean inverse) {
		// TODO Auto-generated method stub
		return z;
	}


	/**
	 * Calculating variance from specified sample.
	 * @param inputSample specified sample.
	 * @return variance from specified sample. This variance is also called 
	 */
	public synchronized double variance(Fetcher<Profile> inputSample) {
		double ss = 0;
		int count = 0;
		
		try {
			while (inputSample.next()) {
				Profile profile = inputSample.pick();
				if (profile == null)
					continue;
				
				double zValue = extractNumber(extractResponseValue(profile));
				double executedValue = extractNumber(execute(profile)); //Synchronize due to execute method.
				if (Util.isUsed(zValue) && Util.isUsed(executedValue)) {
					double d = executedValue - zValue;
					ss += d*d;
					count++;
				}
			}
			inputSample.reset();
			
			if (count == 0)
				return Constants.UNUSED;
			else
				return ss / (double)count;
		}
		catch (Exception e) {
			e.printStackTrace();
		}
		
		return Constants.UNUSED;
	}
	
	
	/**
	 * Estimating given sample.
	 * @param inputSample given sample.
	 * @return estimated sample.
	 */
	public synchronized Fetcher<Profile> fulfill(Fetcher<Profile> inputSample) {
		// TODO Auto-generated method stub
		if (this.getParameter() == null)
			return null;
		
		REMImpl em = (REMImpl)this.newInstance();
		LargeStatistics stat = null;
		try {
			if (em.prepareInternalData(inputSample))
				stat = (LargeStatistics) this.expectation(this.getParameter(), em.getData());
		}
		catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			stat = null;
		}
		if (stat == null)
			return null;
		
		int N = stat.getZData().size();
		AttributeList attRef = getSampleAttributeList(inputSample);
		List<Profile> profiles = Util.newList();
		for (int i = 0; i < N; i++) {
			Profile profile = new Profile(attRef);
			double[] xvector = stat.getXData().get(i);
			double z = stat.getZData().get(i)[1];
			profile.setValue(attRef.size() - 1, z);
			
			for (int j = 0; j < attRef.size() - 1; j++) {
				int foundX = findIndex(em.xIndices, j);
				if (foundX >= 0)
					profile.setValue(j, xvector[foundX]);
			}
			profiles.add(profile);
		}
		em.clearInternalData();

		return new MemFetcher<>(profiles);
	}


    /**
	 * Getting complete data from specified data.
	 * @param data specified data.
	 * @return complete data from specified data.
	 */
	public static LargeStatistics getCompleteData(LargeStatistics data) {
		int N = data.getZData().size();
		List<double[]> xStatistic = Util.newList();
		List<double[]> zStatistic = Util.newList();
		for (int i = 0; i < N; i++) {
			double[] zVector = data.getZData().get(i);
			if (!Util.isUsed(zVector[1]))
				continue;
			
			double[] xVector = data.getXData().get(i);
			boolean missing = false;
			for (int j = 0; j < xVector.length; j++) {
				if (!Util.isUsed(xVector[j])) {
					missing = true;
					break;
				}
			}
			
			if (!missing) {
				xStatistic.add(xVector);
				zStatistic.add(zVector);
			}
		}
		
		if (xStatistic.size() == 0 || zStatistic.size() == 0)
			return null;
		else
			return new LargeStatistics(xStatistic, zStatistic);
	}
	

	/**
	 * Initializing coefficients alpha and betas.
	 * @param regressorNumber the number regressors.
	 * @param random if true, randomization is processed.
	 * @return coefficients alpha and betas.
	 */
	public static ExchangedParameter initializeAlphaBetas(int regressorNumber, boolean random) {
		Random rnd = new Random();
		List<Double> alpha0 = Util.newList(regressorNumber);
		List<double[]> betas0 = Util.newList(regressorNumber);
		for (int j = 0; j < regressorNumber; j++) {
			alpha0.add(random ? rnd.nextDouble() : 0.0);
			
			double[] beta0 = new double[2];
			if (j == 0) {
				beta0[0] = 1;
				beta0[1] = 0;
			}
			else {
				beta0[0] = random ? rnd.nextDouble() : 0.0;
				beta0[1] = random ? rnd.nextDouble() : 0.0;
			}
			betas0.add(beta0);
		}
		
		return new ExchangedParameter(alpha0, betas0);
	}
	
	
	@Override
    public synchronized Graph createRegressorGraph(int xIndex) {
		if (getLargeStatistics() == null || getExchangedParameter() == null)
			return null;
    	
		ExchangedParameter parameter = getExchangedParameter();
		double coeff0 = parameter.getAlpha().get(0);
		double coeff1 = parameter.getAlpha().get(xIndex);
		if (coeff1 == 0) return null;
			
		LargeStatistics stats = getLargeStatistics();
    	int ncurves = 2;
    	int npoints = stats.size();
    	double[][] data = PlotGraph.data(ncurves, npoints);
    	
    	for(int i = 0; i < npoints; i++) {
            data[0][i] = stats.getXData().get(i)[xIndex];
            data[1][i] = stats.getZData().get(i)[1];
        }
    	
    	data[2][0] = Fmath.minimum(data[0]);
    	data[3][0] = coeff0 + coeff1 * data[2][0];
    	data[2][1] = Fmath.maximum(data[0]);
    	data[3][1] = coeff0 + coeff1 * data[2][1];

    	PlotGraphExt pg = new PlotGraphExt(data);

    	pg.setGraphTitle("Regressor plot");
    	pg.setXaxisLegend(extractRegressor(xIndex).toString());
    	pg.setYaxisLegend(extractResponse().toString());
    	int[] popt = {1, 0};
    	pg.setPoint(popt);
    	int[] lopt = {0, 3};
    	pg.setLine(lopt);

    	pg.setBackground(Color.WHITE);
        return pg;
    }

    
	@Override
    public synchronized Graph createResponseGraph() {
		return AbstractRM.createResponseGraph(this, this.getLargeStatistics());
    }
    
    
    @Override
    public synchronized Graph createErrorGraph() {
    	return AbstractRM.createErrorGraph(this, this.getLargeStatistics());
    }

    
    @Override
    public synchronized List<Graph> createResponseRalatedGraphs() {
    	return AbstractRM.createResponseRalatedGraphs(this);
    }
    
    
    @Override
    public synchronized double calcVariance() {
    	return AbstractRM.calcVariance(this, this.getLargeStatistics());
    }
    
    
    @Override
    public synchronized double calcR() {
    	return AbstractRM.calcR(this, this.getLargeStatistics());
    }
    
    
    /**
     * Calculating mean and variance of errors.
     * @return mean and variance of errors.
     */
    public synchronized double[] calcError() {
    	return AbstractRM.calcError(this, this.getLargeStatistics());
    }
    
    
}


