#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <grob/grid_objects.hpp>
#include <span>
#include <ranges>
#include <numbers>
template <typename Array_t,typename FuncType>
inline double IntIndex(Array_t X_arr,FuncType F){
	double sum = 0;
	for(size_t i=0;i<X_arr.size();++i){
		sum += F(i);
	}
	sum -= (0.5*F(0)+0.5*F(X_arr.size()-1));
	return sum*(X_arr[1]-X_arr[0]);
}

template <typename Array_t,typename FuncType>
inline double IntFunc(Array_t X_arr,FuncType F){
	double sum = 0;
	for(size_t i=0;i<X_arr.size();++i){
		sum += F(X_arr[i]);
	}
	sum -= (0.5*F(X_arr[0])+0.5*F(X_arr[X_arr.size()-1]));
	return sum*(X_arr[1]-X_arr[0]);
}

template <typename Funct_t,typename...Arrays_t>
struct VecApply{
	typedef std::invoke_result_t<Funct_t, decltype(std::declval<Arrays_t>()[0])...> type;
};
template <typename Funct_t,typename...Arrays_t>
using VecApply_t = typename VecApply<Funct_t,Arrays_t...>::type;

template <typename Funct_t,typename Array_t,typename...Arrays_t>
std::vector<VecApply_t<Funct_t,Array_t,Arrays_t...>> apply_v(Funct_t const& Func,Array_t const & Vec0,Arrays_t const &... Vecs){
	std::vector<VecApply_t<Funct_t,Array_t,Arrays_t...>> Result(Vec0.size());
	for(size_t i=0;i<Vec0.size();++i){
		Result[i] = Func(Vec0[i],Vecs[i]...);
	}
	return Result;
}

template <typename T>
std::vector<T> linspace(T a,std::type_identity_t<T> b,size_t N,bool incl_last = true){
	if(incl_last)
		return grob::GridVector<T>(a,b,N);
	else{
		return grob::GridVector<T>(a,a + ( (b-a)*N)/(N+1),N);
	}
}
float sinch(float x){
	if(x < 0)
		x = -x;
	if(x < 0.0001){
		return 1 + x*x/6;
	} else {
		return sinh(x)/x;
	}
}

float IntegrateArrays(
	pybind11::array_t<float> RGrid,
	pybind11::array_t<float> PhiGrid, 
	pybind11::array_t<float> DensNuc,
	float U0, float Vesc,float Wmax,float Wdisp,
	float MW,float delta,
	float MN,float nuc_b, 
	size_t delay_i_r,size_t Nu
){
	size_t Nsize = (RGrid.size() -1)/delay_i_r + 1;
	std::vector< float>Rs(Nsize);
	std::vector< float>Phi(Nsize);
	std::vector<float> nr(Nsize);
	for(size_t i=0;i<Nsize;++i){
		Rs[i] = RGrid.data()[i*delay_i_r];
		Phi[i] = PhiGrid.data()[i*delay_i_r];
		nr[i] = DensNuc.data()[i*delay_i_r];
	}

	std::vector<float> VescArr = apply_v([Vesc](float phi){return Vesc*std::sqrt(phi);},Phi);
	
	float mu_k = MW/(MW+MN);
	float mu_p = MN/(MW+MN);
	float mcm = mu_k*MN;
	float v2_delta = 2*delta/mcm;

	constexpr float pi  = std::numbers::pi;
	float SqrW = std::pow(2*pi*Wdisp*Wdisp,-1.5);
	float SigmaSq_inv = 1/(Wdisp*Wdisp);
	

	float NucB2 = nuc_b*nuc_b*MN*MN;
	float Vesc_inv = 1/Vesc;
	return std::pow((MW+0.938f)/(MW+MN),2)*IntIndex(Rs,[&](size_t i)->float{
		float r = Rs[i];
		float vesc = VescArr[i];

		float Umin = std::sqrt (std::max(v2_delta - vesc*vesc,0.0f) );
		grob::GridUniform<float> Uarr(Umin,U0+Wmax,Nu);

		return 3*r*r*nr[i]*IntFunc(Uarr,[&](float u)->float{
			float v = std::sqrt(u*u + vesc*vesc);
			float weight_u = 4*pi*v*u*SqrW*std::exp(-0.5*(u*u+U0*U0)*SigmaSq_inv)*sinch(u*U0*SigmaSq_inv);

			float Vcm = mu_k*v;
			//float dV_in = mu_p*v
			float dv_pk = std::sqrt(v*v - v2_delta);
			float dv_p = mu_k*dv_pk;
			float dv = mu_p*dv_pk;

			float maxCos = (vesc*vesc-Vcm*Vcm-dv*dv)/(2*Vcm*dv);
			if(! (maxCos > -1) ){
				return 0.0f;
			}
			maxCos = std::min(maxCos,1.0f);
			grob::GridUniform<float> cosArray(-1,maxCos,101);
			return (dv_pk*Vesc_inv)*weight_u/2*IntFunc(cosArray,[&](float cosTheta)->float{
				float v2_p = Vcm*Vcm + dv_p*dv_p - 2*Vcm*dv_p*cosTheta;
				float y = NucB2*v2_p/2;
				return exp(-y);
			});
		});
	});
}


PYBIND11_MODULE(Integrate,m){
	namespace py = pybind11;
	m.def("Capture",IntegrateArrays,
		"",
		py::arg("Radius"),
		py::arg("Phi"),
		py::arg("DenseNuc"),
		py::arg("U0"),
		py::arg("Vesc"),
		py::arg("Wmax"),
		py::arg("Wdisp"),
		py::arg("m_wimp"),
		py::arg("delta"),
		py::arg("m_nuc"),
		py::arg("b"),
		py::arg("dN_r"),
		py::arg("Nu")
	);
}