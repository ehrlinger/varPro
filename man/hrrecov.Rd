\name{hrrecov}
\docType{data}
\alias{hrrecov}
\title{Exercise Heart Rate Recovery and Mortality}
\description{
A survival dataset from a cohort of patients referred for symptom-limited exercise testing,
originally used to study exercise heart rate recovery as a predictor of all-cause mortality.
}
\usage{
data(hrrecov)
}
\format{
A data frame with 23701 observations on 83 variables.
\describe{
  \item{\code{ttodead}}{Follow-up time to death or censoring, in years (rounded to 2 decimals).}
  \item{\code{died}}{Event indicator (1 = death, 0 = right-censored).}
  \item{\code{hrrecov}}{Heart rate recovery (beats/min): peak heart rate minus heart rate 1 minute into recovery.}
  \item{\code{lowrec}}{Indicator for low/abnormal heart rate recovery.}
  \item{\code{lowcri}}{Indicator for low chronotropic response index/chronotropic incompetence.}
  \item{\code{peak_hr}}{Peak heart rate during exercise (beats/min).}
  \item{\code{peak_met}}{Peak workload achieved (metabolic equivalents, METs).}
  \item{\code{fitness}}{Fitness category (ordinal 1-5; coding as in source data).}
  \item{\code{heart_ra}}{Resting heart rate prior to testing (beats/min).}
  \item{\code{sbprest}}{Resting systolic blood pressure (mm Hg).}
  \item{\code{dbprest}}{Resting diastolic blood pressure (mm Hg).}
  \item{\code{age}}{Age at exercise test (years).}
  \item{\code{gender}}{Sex indicator (1=men).}
  \item{\code{race}}{Race category (integer code; coding as in source data).}
  \item{\code{black}}{Indicator for Black race (0/1).}
  \item{\code{height}}{Height (m).}
  \item{\code{weight}}{Weight (kg).}
  \item{\code{bmi}}{Body mass index (kg/m^2).}
  \item{\code{bsa}}{Body surface area.}
  \item{\code{wtht}}{Weight-to-height ratio (weight/height; kg/m).}
  \item{\code{obese}}{Indicator for obesity (0/1).}
  \item{\code{priorcad}}{Known or suspected coronary artery disease prior to test (0/1).}
  \item{\code{mihist}}{History of myocardial infarction (0/1).}
  \item{\code{pcabg}}{Prior coronary artery bypass grafting (CABG) (0/1).}
  \item{\code{ppci}}{Prior percutaneous coronary intervention (PCI) (0/1).}
  \item{\code{cva}}{History of cerebrovascular accident/stroke (0/1).}
  \item{\code{tia}}{History of transient ischemic attack (0/1).}
  \item{\code{pvd}}{Peripheral vascular disease (0/1).}
  \item{\code{diabetes}}{History of diabetes mellitus (0/1).}
  \item{\code{insulin}}{Insulin therapy (0/1).}
  \item{\code{htn}}{History of hypertension (0/1).}
  \item{\code{htnrx}}{Antihypertensive treatment (0/1).}
  \item{\code{hichol}}{History of high cholesterol/hyperlipidemia (0/1).}
  \item{\code{smknow}}{Smoking history/status indicator.}
  \item{\code{asthma}}{History of asthma (0/1).}
  \item{\code{copd}}{History of chronic obstructive pulmonary disease (0/1).}
  \item{\code{esrd}}{End-stage renal disease (0/1).}
  \item{\code{lv_dysf}}{Left ventricular dysfunction (0/1).}
  \item{\code{ecgmi}}{ECG evidence of myocardial infarction (0/1).}
  \item{\code{ecglvh}}{ECG evidence of left ventricular hypertrophy (0/1).}
  \item{\code{lbbb}}{Left bundle branch block (0/1).}
  \item{\code{rbbb}}{Right bundle branch block (0/1).}
  \item{\code{restst}}{Resting ST-segment abnormality (0/1).}
  \item{\code{stnond}}{Non-diagnostic ST-segment response (0/1).}
  \item{\code{stabn}}{ST-segment abnormality during testing (0/1).}
  \item{\code{stabnb}}{ST-segment abnormality subtype/flag.}
  \item{\code{stabnv}}{ST-segment abnormality subtype/flag.}
  \item{\code{rtachy}}{Tachycardia indicator.}
  \item{\code{typcp}}{Typical chest pain indicator.}
  \item{\code{ntangina}}{Angina history/symptom indicator.}
  \item{\code{ttangina}}{Angina during treadmill test (0/1).}
  \item{\code{ttclaud}}{Claudication during treadmill test (0/1).}
  \item{\code{image}}{Exercise test performed with imaging (0/1).}
  \item{\code{acei}}{Angiotensin-converting enzyme (ACE) inhibitor use (0/1).}
  \item{\code{aspirin}}{Aspirin use (0/1).}
  \item{\code{betablok}}{Beta-blocker use (0/1).}
  \item{\code{dilver}}{Diltiazem/verapamil use (0/1).}
  \item{\code{nifed}}{Nifedipine use (0/1).}
  \item{\code{diuretic}}{Diuretic use (0/1).}
  \item{\code{lipidrx}}{Lipid-lowering therapy use (0/1).}
  \item{\code{nitrates}}{Nitrate therapy use (0/1).}
  \item{\code{bdilat}}{Bronchodilator use (0/1).}
  \item{\code{rs_ami}}{Reason for referral/testing: acute myocardial infarction (0/1).}
  \item{\code{rs_mi}}{Reason for referral/testing: myocardial infarction (0/1).}
  \item{\code{rs_ptca}}{Reason for referral/testing: PTCA/angioplasty (0/1).}
  \item{\code{rs_cabg}}{Reason for referral/testing: CABG (0/1).}
  \item{\code{rs_arrth}}{Reason for referral/testing: arrhythmia (0/1).}
  \item{\code{rs_htran}}{Reason for referral/testing: heart transplant (0/1).}
  \item{\code{bpvc_rst}}{Bigeminal premature ventricular contractions at rest (0/1).}
  \item{\code{bpvc_ex}}{Bigeminal premature ventricular contractions during exercise (0/1).}
  \item{\code{bpvc_rec}}{Bigeminal premature ventricular contractions during recovery (0/1).}
  \item{\code{fpvc_rst}}{Frequent premature ventricular contractions at rest (0/1).}
  \item{\code{fpvc_ex}}{Frequent premature ventricular contractions during exercise (0/1).}
  \item{\code{fpvc_rec}}{Frequent premature ventricular contractions during recovery (0/1).}
  \item{\code{nsvt_rst}}{Non-sustained ventricular tachycardia at rest (0/1).}
  \item{\code{nsvt_ex}}{Non-sustained ventricular tachycardia during exercise (0/1).}
  \item{\code{nsvt_rec}}{Non-sustained ventricular tachycardia during recovery (0/1).}
  \item{\code{vtrp_rst}}{Ventricular triplets at rest (0/1).}
  \item{\code{vtrp_ex}}{Ventricular triplets during exercise (0/1).}
  \item{\code{vtrp_rec}}{Ventricular triplets during recovery (0/1).}
  \item{\code{hbm2_rec}}{Heart rate-related recovery measure at 2 minutes.}
  \item{\code{exec}}{Indicator variable (0/1); coding as in source data.}
  \item{\code{aso}}{Indicator variable (0/1); coding as in source data.}
}
}
\details{
Heart rate recovery (\code{hrrecov}) is defined as peak heart rate minus the heart rate
measured 1 minute into recovery. The survival outcome is all-cause mortality with right-censoring.
Unless otherwise noted, indicator variables are coded 0/1 (0 = no/absent, 1 = yes/present).
}
\references{
Ishwaran, H., Blackstone, E. H., Pothier, C. E. and Lauer, M. S. (2004).
Relative risk forests for exercise heart rate recovery as a predictor of mortality.
\emph{Journal of the American Statistical Association}, 99, 591-600.
}
\examples{
\donttest{
data(hrrecov)
vp <- varpro(Surv(ttodead, died)~., hrrecov, ntree=50, split.weight=FALSE) 
ivp <- ivarpro(vp)
partial.ivarpro(ivp, "peak_met", "fitness", "peak_hr",
        col.legend.n = 7, smooth.n = 7, x.dist = "auto")
}
}
\keyword{datasets}