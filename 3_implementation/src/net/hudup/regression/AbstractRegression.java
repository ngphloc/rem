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
	protected List<Integer> xIndices = new ArrayList<>();

	
	/**
	 * Indices for Z data.
	 */
	protected List<Integer> zIndices = new ArrayList<>();
	
	
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
	public synchronized void unsetup() {
		super.unsetup();
		
		this.coeffs = null;
		this.xIndices.clear();
		this.zIndices.clear();
		this.attList = null;
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
			double value = profile.getValueAsReal(this.xIndices.get(j + 1)); //due to x = (1, x1, x2,..., xn) and xIndices.get(0) = -1
			sum += this.coeffs[j + 1] * value; 
		}
		
		return sum;
	}

	
	@Override
	public synchronized Object getParameter() {
		// TODO Auto-generated method stub
		return coeffs;
	}

	
	@Override
	public int getResponseIndex() {
		// TODO Auto-generated method stub
		return zIndices.get(1);
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
		buffer.append(this.attList.get(this.zIndices.get(this.zIndices.size() - 1)).getName()
				+ " = " + MathUtil.format(coeffs[0]));
		for (int j = 0; j < this.coeffs.length - 1; j++) {
			double coeff = this.coeffs[j + 1];
			String variableName = this.attList.get(this.xIndices.get(j + 1)).getName();
			if (coeff < 0)
				buffer.append(" - " + MathUtil.format(Math.abs(coeff)) + "*" + variableName);
			else
				buffer.append(" + " + MathUtil.format(coeff) + "*" + variableName);
		}
		
		return buffer.toString();
	}

	
}
