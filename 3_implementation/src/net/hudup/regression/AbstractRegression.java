package net.hudup.regression;

import java.util.ArrayList;
import java.util.List;

import net.hudup.core.alg.AbstractTestingAlg;
import net.hudup.core.data.AttributeList;
import net.hudup.core.data.DataConfig;
import net.hudup.core.data.Profile;
import net.hudup.core.logistic.MathUtil;

/**
 * <code>AbstractRegression</code> is the most abstract class for expectation maximization (EM) algorithm.
 * It implements partially the interface {@link Regression}.
 * 
 * @author Loc Nguyen
 * @version 1.0*
 */
public abstract class AbstractRegression extends AbstractTestingAlg implements Regression {

	
	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Regression indices field.
	 */
	protected final static String INDICES_FIELD = "regress_indices";

	
	/**
	 * Regression coefficient.
	 */
	protected double[] coeffs = null;
	
	
	/**
	 * Indices for X data.
	 */
	protected List<int[]> xIndices = new ArrayList<>();

	
	/**
	 * Indices for Z data.
	 */
	protected List<int[]> zIndices = new ArrayList<>();
	
	
	/**
	 * Attribute list for all variables: all X, Y, and z.
	 */
	protected AttributeList attList = null;
	

    /**
     * Default constructor
     */
	public AbstractRegression() {
		super();
		// TODO Auto-generated constructor stub
	}


	@Override
	public synchronized Object execute(Object input) {
		// TODO Auto-generated method stub
		if (this.coeffs == null)
			return null;
		
		if (input == null || !(input instanceof Profile))
			return null; //only support profile input currently
		Profile profile = (Profile)input;
		
		double sum = this.coeffs[0];
		for (int j= 0; j < this.coeffs.length - 1; j++) {
			double value = extractRegressor(profile, j + 1); //due to x = (1, x1, x2,..., xn) and xIndices.get(0) = -1
			sum += this.coeffs[j + 1] * (double)transformRegressor(value); 
		}
		
		return inverseTransformResponse(sum);
	}

	
	@Override
	public synchronized Object getParameter() {
		// TODO Auto-generated method stub
		return coeffs;
	}

	
	@Override
	public DataConfig createDefaultConfig() {
		// TODO Auto-generated method stub
		DataConfig config = super.createDefaultConfig();
		config.put(INDICES_FIELD, "-1, -1, -1"); //Not used
		return config;
	}


	@Override
	public String parameterToShownText(Object parameter, Object... info) {
		// TODO Auto-generated method stub
		if (parameter == null || !(parameter instanceof double[]))
			return "";
		double[] coeffs = (double[])parameter;

		StringBuffer buffer = new StringBuffer();
		for (int j = 0; j < coeffs.length; j++) {
			if (j > 0)
				buffer.append(", ");
			buffer.append(MathUtil.format(coeffs[j]));
		}
		
		return buffer.toString();
	}

	
	@Override
	public synchronized String getDescription() {
		// TODO Auto-generated method stub
		if (this.coeffs == null)
			return "";
		
		StringBuffer buffer = new StringBuffer();
		buffer.append(transformResponse(extractResponseText()) + " = " + MathUtil.format(coeffs[0]));
		for (int j = 0; j < this.coeffs.length - 1; j++) {
			double coeff = this.coeffs[j + 1];
			String variableName = transformRegressor(extractRegressorText(j + 1)).toString();
			if (coeff < 0)
				buffer.append(" - " + MathUtil.format(Math.abs(coeff)) + "*" + variableName);
			else
				buffer.append(" + " + MathUtil.format(coeff) + "*" + variableName);
		}
		
		return buffer.toString();
	}
	
	
	/**
	 * Extracting value of regressor (X) from specified profile.
	 * @param profile specified profile.
	 * @param index specified indices.
	 * @return value of regressor (X) extracted from specified profile.
	 */
	protected double extractRegressor(Profile profile, int index) {
		// TODO Auto-generated method stub
		return profile.getValueAsReal(xIndices.get(index)[0]);
	}


	/**
	 * Extracting text of regressor (X).
	 * @param index specified indices.
	 * @return text of regressor (X) extracted.
	 */
	protected String extractRegressorText(int index) {
		// TODO Auto-generated method stub
		return attList.get(xIndices.get(index)[0]).getName();
	}


	@Override
	public double extractResponse(Profile profile) {
		// TODO Auto-generated method stub
		return profile.getValueAsReal(zIndices.get(1)[0]);
	}


	/**
	 * Extracting text of response variable (Z).
	 * @return text of response variable (Z) extracted.
	 */
	protected String extractResponseText() {
		// TODO Auto-generated method stub
		return attList.get(zIndices.get(1)[0]).getName();
	}


	/**
	 * Transforming independent variable X.
	 * @param x specified variable X.
	 * @return transformed value of X.
	 */
	protected Object transformRegressor(Object x) {
		// TODO Auto-generated method stub
		if (x == null)
			return null;
		else if (x instanceof Number)
			return ((Number)x).doubleValue();
		else if (x instanceof String)
			return (String)x;
		else
			return x;
	}


	/**
	 * Inverse transforming of the inverse value of independent variable X.
	 * This method is the inverse of {@link #transformRegressor(double)}.
	 * @param inverseX inverse value of independent variable X.
	 * @return value of X.
	 */
	protected Object inverseTransformRegressor(Object inverseX) {
		// TODO Auto-generated method stub
		return transformRegressor(inverseX);
	}


	/**
	 * Transforming independent variable Z.
	 * @param z specified variable Z.
	 * @return transformed value of Z.
	 */
	protected Object transformResponse(Object z) {
		// TODO Auto-generated method stub
		if (z == null)
			return null;
		else if (z instanceof Number)
			return ((Number)z).doubleValue();
		else if (z instanceof String)
			return (String)z;
		else
			return z;
	}


	/**
	 * Inverse transforming of the inverse value of independent variable Z.
	 * This method is the inverse of {@link #transformResponse(double)}.
	 * @param inverseZ inverse value of independent variable Z.
	 * @return value of Z.
	 */
	protected Object inverseTransformResponse(Object inverseZ) {
		// TODO Auto-generated method stub
		return transformResponse(inverseZ);
	}

	
}
