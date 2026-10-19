import os
import requests
import json
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.document_loader import download_sample_kcc_data

RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
os.makedirs(RAW_DATA_DIR, exist_ok=True)


def download_public_advisories():
    """Download or create 3 public agricultural advisory text files."""
    print("Fetching / Creating public agricultural advisories...")
    
    advisory_1_content = """# National Agro-Meteorological Advisory Bulletin - Kharif & Rabi Transition

The India Meteorological Department (IMD) and Agromet Advisory Service issue regular crop recommendations based on prevailing weather patterns across Indian agro-climatic zones.

## General Weather Preparedness
Farmers across Northern and Central India should monitor soil moisture levels following post-monsoon withdrawal. In areas experiencing isolated pre-winter rain showers, postpone intercultural operations such as hoeing and weeding for 48 hours to prevent soil compaction.

## Sowing Directives for Rabi Crops
Optimal sowing window for Rabi crops like wheat, chickpea, mustard, and barley begins when mean daily temperatures drop consistently below 25°C. Sowing in warm soil hampers germination and seedling vigour. Treat seeds before sowing with Trichoderma viride @ 5g/kg or Captan @ 2.5g/kg seed to prevent seedling blight and collar rot.

## Pest and Disease Vigilance
In humid regions of Gangetic plains, early morning dew creates microclimatic conditions conducive to powdery mildew and aphid infestation. Farmers are advised to set up yellow sticky traps (10 per acre) and survey fields twice a week. If aphid density exceeds economic threshold levels (ETL), spray neem seed kernel extract (NSKE 5%) or recommended systemic insecticides during late afternoon hours.
"""

    advisory_2_content = """# Comprehensive Pest and Soil Health Advisory - Monsoon & Post-Monsoon Management

Soil health and nutrient balance represent the foundation of sustainable agricultural yields under changing climatic extremes.

## Integrated Nutrient Management (INM)
Continuous indiscriminate application of nitrogenous fertilizers without micronutrients degrades soil biological activity. Farmers should incorporate farmyard manure (FYM) or well-rotted compost @ 5-10 tonnes/hectare during land preparation. Soil testing should be carried out every two years to identify deficiencies in Zinc, Iron, and Boron.

## Integrated Pest Management Strategies
To mitigate pest resistance and minimize chemical residues, adopt mechanical and biological controls before chemical interventions. Pheromone traps for Spodoptera litura and Helicoverpa armigera @ 5 traps/acre serve as effective monitoring indicators. For leaf-eating caterpillars, spray Bacillus thuringiensis (Bt) formulations @ 1 kg/ha or apply nuclear polyhedrosis virus (NPV) when larvae are young.

## Waterlogging and Drainage Mitigation
Heavy and unseasonal rainfall can lead to root asphyxiation in pulses and vegetables. Farmers should excavate field boundary drainage channels with minimum 0.5% slope to evacuate standing water within 24 hours of heavy precipitation events.
"""

    advisory_3_content = """# State Extension Advisory: Climate Resilient Cropping Systems

State agricultural departments and Krishi Vigyan Kendras (KVKs) recommend adoption of climate-smart farming techniques to manage erratic weather variations.

## Micro-Irrigation Technologies
Drip and sprinkler irrigation technologies increase water application efficiency up to 85-90% compared to traditional flood irrigation. Farmers cultivating high-value horticultural crops and cotton should connect drip systems with fertigation units to supply soluble NPK nutrients directly to root zones.

## Intercropping and Risk Diversification
Monoculture increases vulnerability to crop failure during pest outbreaks or rainfall deficits. Planting pigeonpea with soybean (2:4 ratio) or groundnut with pearl millet (3:1 ratio) provides safety buffers against total economic loss while naturally enriching soil nitrogen through biological nitrogen fixation.

## Post-Harvest Grain Moisture Control
Grains harvested during humid weather must be thoroughly sun-dried on tarpaulins to avoid moisture absorption from wet ground. Grain moisture content must be below 10-12% before bagging in triple-layer hermetic bags (Purdue Improved Crop Storage bags) to eliminate storage pest infestation without chemical fumigation.
"""

    files = [
        ("advisory_bulletin_imd.txt", advisory_1_content),
        ("advisory_soil_pest_management.txt", advisory_2_content),
        ("advisory_climate_resilient_farming.txt", advisory_3_content)
    ]
    for filename, content in files:
        filepath = os.path.join(RAW_DATA_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content.strip() + "\n")
        print(f"Saved advisory: {filepath}")


def create_icar_advisories():
    """Create 3 detailed ICAR-style advisory text files (800-1000 words each)."""
    print("Generating ICAR-style detailed advisories...")

    # 1. Wheat Disease Management
    wheat_advisory = """# ICAR Technical Advisory Bulletin: Comprehensive Disease Management in Wheat Cultivation

## 1. Introduction and Economic Importance
Wheat (Triticum aestivum) is the premier rabi cereal crop of India, occupying over 30 million hectares predominantly in the Indo-Gangetic Plains. The productivity of wheat is substantially challenged by several fungal pathogens, abiotic stress factors, and climate variations. Timely diagnosis, prophylactic measures, and integrated chemical applications are imperative to safeguard farm incomes and national food security.

## 2. Major Fungal Diseases of Wheat

### A. Yellow (Stripe) Rust (Puccinia striiformis f. sp. tritici)
Yellow rust is a devastating foliar disease prevalent in the North-Western Plains Zone (Punjab, Haryana, Western Uttar Pradesh, Jammu & Kashmir, and Himachal Pradesh).
- **Symptoms:** Yellow pustules arranged in characteristic linear stripes parallel to leaf veins, releasing yellow urediniospores when touched. In severe conditions, heads, glumes, and awns become infected.
- **Predisposing Conditions:** Temperatures between 10°C to 20°C combined with high relative humidity (>85%), frequent morning fog, and cloudy weather during January and February.
- **Control Measures:** Cultivate resistant cultivars such as HD-3086, DBW-187, DBW-222, and PBW-550. If yellow pustules appear, spray Propiconazole 25% EC (Tilt) @ 0.1% (500 ml dissolved in 500 litres of water per hectare) or Tebuconazole 250 EC @ 1 ml/litre. Repeat after 15 days if conditions remain cloudy and wet.

### B. Brown (Leaf) Rust (Puccinia triticina)
Brown rust occurs across all wheat zones towards late rabi (February-March).
- **Symptoms:** Small, round to oblong orange-brown pustules scattered irregularly over the leaf lamina without forming parallel lines.
- **Predisposing Factors:** Warm temperatures (20°C to 25°C) with free moisture on leaf blades.
- **Control Measures:** Grow tolerant varieties like HD-2967, WH-1105, or GW-322. Spray Mancozeb 75 WP @ 2.5 g/litre or Hexaconazole 5 EC @ 1 ml/litre at early onset.

### C. Karnal Bunt (Tilletia indica)
Karnal bunt is an internationally quarantined disease that impairs grain quality and flour aroma.
- **Symptoms:** Only a portion of each individual grain is transformed into a black sooty powdery mass of teliospores smelling of rotten fish due to trimethylamine emission.
- **Control Measures:** Avoid sowing infected seed stocks. Seed treatment with Thiram 75 WP or Carbendazim 50 WP @ 2 g/kg seed. Avoid heavy flood irrigation during the boot to anthesis stages. Spray Propiconazole 25 EC @ 500 ml/ha during 50% ear emergence stage if disease history exists.

### D. Loose Smut (Ustilago tritici)
An internally seed-borne pathogen where every floral part is replaced by an olive-black teliospore mass.
- **Control Measures:** Hot water seed treatment (soaking seeds in water at 20°C for 5 hours followed by 52°C water bath for 10 minutes) or systemic chemical seed treatment with Carboxin 75 WP or Tebuconazole 2 DS @ 1.5 g/kg seed.

## 3. Cultural Practices and Preventative Agronomy
1. **Crop Rotation:** Rotate wheat with leguminous crops like chickpea or lentil to break pathogen lifecycles.
2. **Balanced Nutrition:** Apply NPK fertilizers strictly in the ratio of 120:60:40 kg/ha. Avoid excess split nitrogen doses during late vegetative growth, as excessive succulent vegetative tissue predisposes the crop to rust.
3. **Optimized Seed Density:** Maintain seed rate of 100 kg/ha for timely sown irrigated crop and 125 kg/ha for late sown conditions using seed drills at 20-22.5 cm row spacing.
4. **Water Drainage:** Prevent water stagnation during initial crown root initiation and grain filling stages.
"""

    # 2. Rice Cultivation Practices
    rice_advisory = """# ICAR Technical Guide: Modern Rice Cultivation Practices, Pest Control, and Water Management

## 1. Agro-Ecological Context
Rice (Oryza sativa) constitutes the staple food for more than 60 percent of the Indian populace. Sustainable intensification through the System of Rice Intensification (SRI), Direct Seeded Rice (DSR), and scientific transplanted puddled rice helps optimize water productivity and minimizes greenhouse emissions while boosting paddy yields.

## 2. Nursery Management and Land Preparation
For conventional transplanted rice, establish raised nursery beds 1.5 m wide with drainage channels. Use 25-30 kg certified seed per hectare.
- **Seed Treatment:** Soak seeds in brine solution (10% salt) to eliminate chaffy lightweight seeds. Wash viable sunk seeds and soak for 24 hours in Streptocycline solution (0.1 g/litre) plus Carbendazim @ 2 g/kg seed to eradicate seed-borne bacterial blight and brown spot.
- **Main Field Puddling:** Puddle soil twice with disc harrows followed by leveling. Incorporate 5 tonnes of well-decomposed FYM or compost per hectare at primary tillage.

## 3. Transplanting and Sowing Protocols
- **Transplanting Window:** Transplant 21-25 day old seedlings (3-4 leaves stage) at 2-3 seedlings per hill with 20 cm x 15 cm spacing.
- **Direct Seeded Rice (DSR):** Sown using zero-till seed-cum-fertilizer drills before onset of heavy monsoon showers. Treat seeds with fungicides and apply pre-emergence herbicide Pendimethalin 30 EC @ 1.0 kg a.i./ha within 48 hours of sowing to keep fields weed-free.

## 4. Water Management: Alternate Wetting and Drying (AWD)
Continuous ponding of 5-10 cm water wastes enormous underground aquifers and breeds anaerobic methane.
- Maintain 2-3 cm shallow water layer for the first 15 days to assist root anchor and tiller initiation.
- Afterwards, practice Alternate Wetting and Drying (AWD) using perforated field water pipes: allow water table to drop 15 cm below soil surface before re-irrigating to 5 cm depth.
- Drain standing water completely 10-12 days prior to expected harvest to facilitate machine harvesting and uniform grain ripening.

## 5. Major Insect Pests and Management

### A. Yellow Stem Borer (Scirpophaga incertulas)
- **Damage:** Causes "dead hearts" in vegetative phase and "white ears" with chaffy grains during panicle emergence.
- **Control:** Install sex pheromone traps @ 8 traps/ha. Apply Cartap Hydrochloride 4G @ 10 kg/acre or spray Chlorantraniliprole 18.5 SC @ 60 ml/acre in 200 litres water.

### B. Brown Planthopper (BPH - Nilaparvata lugens)
- **Damage:** Sucks sap at base of tillers causing circular dry patches known as "hopper burn".
- **Control:** Provide alleyways of 30 cm every 2-3 meters for aeration. Avoid resurgence-inducing pyrethroids. Spray Triflumezopyrim 10 SC @ 94 ml/acre or Pymetrozine 50 WG @ 120 g/acre directing spray nozzle towards the base of rice clumps.

### C. Rice Blast (Magnaporthe oryzae)
- **Symptoms:** Spindle-shaped lesions with ash-grey centers on leaves (leaf blast) and dark lesions around the neck of panicle (neck blast).
- **Control:** Spray Tricyclazole 75 WP @ 0.6 g/litre or Isoprothiolane 40 EC @ 1.5 ml/litre at panicle initiation stage.
"""

    # 3. Cotton Pest Control
    cotton_advisory = """# ICAR - CICR Advisory: Integrated Pest Management and Canopy Architecture in Cotton Cultivation

## 1. Agro-Ecological Importance
Cotton (Gossypium hirsutum / arboreum) is known as the "White Gold" of India, sustaining millions of farm households across Gujarat, Maharashtra, Telangana, and Punjab. High susceptibility to sucking pests and bollworm complexes requires an integrated pest management (IPM) regime emphasizing biological control, pest phenology, and threshold-based chemical application.

## 2. Early Season Sucking Pest Complex

### A. Cotton Aphids (Aphis gossypii) & Jassids/Leafhoppers (Amrasca biguttula)
- **Symptoms:** Aphids excrete honeydew attracting sooty mould; leafhoppers cause downward curling of leaf margins, yellowing ("hopperburn"), and tissue necrosis.
- **Economic Threshold Levels (ETL):** 2 jassids or 5 aphids per leaf on top, middle, and bottom leaves.
- **Management:**
  1. Seed treatment with Imidacloprid 70 WS @ 7 g/kg or Thiamethoxam 70 WS @ 4 g/kg seed.
  2. Install yellow sticky traps (10-15 per acre) across field borders.
  3. Spray Flonicamid 50 WG @ 80 g/acre or Dinotefuran 20 SG @ 60 g/acre. Avoid early-season broad-spectrum synthetic pyrethroids.

### B. Whitefly (Bemisia tabaci) and Cotton Leaf Curl Virus (CLCuV)
- **Symptoms:** Chlorotic spots, leaf curling, vectoring devastating geminivirus.
- **Management:** Eradicate alternative weed hosts (Abutilon indicum, Parthenium hysterophorus). Spray Neem formulation (Azadirachtin 1500 ppm @ 5 ml/litre) at first notice. For heavy infestations, apply Pyriproxyfen 10 EC @ 400 ml/acre or Spiromesifen 22.9 SC @ 200 ml/acre.

### C. Thrips (Thrips tabaci)
- **Symptoms:** Silvery sheen on underside of leaves, upward cupping of leaves.
- **Management:** Spray Fipronil 5 SC @ 2 ml/litre or Spinetoram 11.7 SC @ 1 ml/litre.

## 3. Bollworm Complex Management

### A. Pink Bollworm (Pectinophora gossypiella)
- **Symptoms:** Rosetted flowers caused by larval webbing of petals; internal boll damage with stained lint and bored seeds.
- **ETL:** 8 moths/trap/night for 3 consecutive days or 10% damaged green bolls with live larvae.
- **Control:**
  1. Mass trapping using gossyplure pheromone traps @ 10 traps/acre.
  2. Release Trichogramma bactrae egg parasitoids @ 60,000/acre at weekly intervals starting 45 days after sowing.
  3. Spray Emamectin Benzoate 5 SG @ 100 g/acre or Chlorantraniliprole 18.5 SC @ 60 ml/acre when ETL is crossed.

### B. American Bollworm (Helicoverpa armigera)
- **Control:** Install bird perches (15-20 per acre), apply HaNPV (Helicoverpa Nuclear Polyhedrosis Virus) @ 250 LE/ha with 0.1% teepol and 0.5% jaggery during evening hours.

## 4. Canopy and Nutrient Management
Excessive vegetative growth reduces boll retention and provides shade that fosters fungal boll rots.
- Spray growth regulator Mepiquat Chloride @ 100 ppm at 70-80 days after sowing to suppress vegetative growth and redirect carbohydrates to developing bolls.
- Foliar application of 2% Potassium Nitrate (13:0:45) at peak flowering and boll development stages substantially improves boll weight and fiber tensile strength.
"""

    icar_files = [
        ("icar_wheat_disease_management.txt", wheat_advisory),
        ("icar_rice_cultivation_and_pests.txt", rice_advisory),
        ("icar_cotton_pest_management.txt", cotton_advisory)
    ]
    for filename, content in icar_files:
        filepath = os.path.join(RAW_DATA_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content.strip() + "\n")
        print(f"Saved ICAR advisory: {filepath}")


def create_government_schemes():
    """Create 2 detailed government scheme files."""
    print("Generating Government scheme documentations...")

    # 1. PM-KISAN
    pm_kisan_content = """# Pradhan Mantri Kisan Samman Nidhi (PM-KISAN) Scheme: Operational Guidelines & Eligibility

## 1. Scheme Overview
Pradhan Mantri Kisan Samman Nidhi (PM-KISAN) is a Central Sector Scheme launched by the Government of India in February 2019 to provide income support to all landholding farmer families across the nation.

## 2. Financial Benefit and Disbursement Mechanism
Under the PM-KISAN scheme, eligible farmer families receive a financial benefit of Rs 6,000 per year.
- The amount is disbursed in three equal four-monthly installments of Rs 2,000 each.
- Transfer periods: April-July (Period 1), August-November (Period 2), and December-March (Period 3).
- Direct Benefit Transfer (DBT): Payments are credited directly into bank accounts linked with Aadhaar and National Payments Corporation of India (NPCI) payment gateway.

## 3. Eligibility Criteria
1. The scheme defines a farmer family as a husband, wife, and minor children owning cultivable land as per land records of the respective State or Union Territory.
2. The beneficiary must possess valid landownership documentation (RoR/Khasra/Khatauni).
3. The landholding must be cultivable agricultural land registered before the cutoff date of February 1, 2019.

## 4. Exclusion Categories
The following categories of beneficiaries are strictly excluded from receiving financial benefits under PM-KISAN:
1. Institutional landholders (agricultural land owned by trusts, religious institutions, corporate farms, or government departments).
2. Farmer families where one or more members belong to the following categories:
   - Former and present holders of constitutional posts.
   - Former and present Ministers, State Ministers, Members of Lok Sabha, Rajya Sabha, State Legislative Assemblies, or Mayors of Municipal Corporations.
   - All serving or retired officers and employees of Central/State Government Ministries, autonomous institutions, and regular employees of local bodies (excluding Multi-Tasking Staff / Class IV / Group D employees).
   - All superannuated/retired pensioners whose monthly pension is Rs 10,000 or more.
   - Persons who paid Income Tax in the last assessment year.
   - Professionals such as Doctors, Engineers, Lawyers, Chartered Accountants, and Architects registered with professional bodies.

## 5. Mandatory Verification and e-KYC Protocol
To ensure targeted delivery and eliminate duplicate entries:
- Aadhaar mandatory linkage: Every beneficiary must complete biometric e-KYC or OTP-based e-KYC on the PM-KISAN portal (pmkisan.gov.in) or mobile application.
- Land record seeding: State revenue departments must verify and seed land ownership details with the national portal.
- Bank Account NPCI DBT-enablement: Beneficiary accounts must have Aadhaar seeding enabled at bank branches.
"""

    # 2. PMFBY
    pmfby_content = """# Pradhan Mantri Fasal Bima Yojana (PMFBY): Comprehensive Scheme Guidelines

## 1. Objectives and Scope
Pradhan Mantri Fasal Bima Yojana (PMFBY) was launched by the Ministry of Agriculture and Farmers Welfare to provide comprehensive financial insurance protection against crop loss resulting from non-preventable natural risks, thereby stabilizing farmer incomes and encouraging modern agricultural investments.

## 2. Crops Covered
- Food crops (Cereals, Millets, and Pulses)
- Oilseeds
- Annual Commercial and Annual Horticultural crops notified by respective State Governments.

## 3. Actuarial Premium Rates Payable by Farmers
PMFBY drastically minimizes premium burden on farmers:
- **Kharif Crops:** Maximum 2.0% of Sum Insured or actuarial rate, whichever is lower.
- **Rabi Food Grain & Oilseed Crops:** Maximum 1.5% of Sum Insured or actuarial rate, whichever is lower.
- **Commercial / Horticultural Crops:** Maximum 5.0% of Sum Insured.
- The balance actuarial premium subsidy is shared 50:50 between the Central Government and State Governments (90:10 for North Eastern states).

## 4. Coverage of Natural Risks and Stages of Cultivation
PMFBY offers end-to-end risk cover throughout the crop cycle:
1. **Prevented Sowing / Planting Risk:** If insured farmers are prevented from sowing or planting due to deficit rainfall or adverse seasonal conditions (up to 25% of sum insured paid).
2. **Standing Crop (Sowing to Harvesting):** Comprehensive risk insurance against non-preventable risks: drought, dry spells, flood, inundation, wide-spread pests and diseases, landslides, natural fire, lightning, storm, cyclone, and hailstorm.
3. **Post-Harvest Losses:** Coverage available up to a maximum period of 14 days from harvesting for crops kept in 'cut and spread' condition in the field against specific perils of hail, cyclone, cyclonic rains, and unseasonal rains.
4. **Localized Calamities:** Loss or damage resulting from occurrence of localized perils like hailstorm, landslide, cloudburst, inundation affecting isolated farms.

## 5. Claim Assessment and Settlement
- Assessment relies on Crop Cutting Experiments (CCEs) conducted under state supervision, increasingly monitored using satellite imagery, remote sensing, and the CROPIC mobile app.
- For localized calamities and post-harvest losses, farmers must intimate the insurance company, local agriculture office, or the toll-free PMFBY helpline (14447) within 72 hours of damage occurrence.
- Essential Documents Required: Aadhaar card, Bank Passbook, Land Revenue Record (Khasra/7-12), Sowing Declaration Certificate, and Tenant Farming Lease Agreement if applicable.
"""

    scheme_files = [
        ("pm_kisan_eligibility_and_guidelines.txt", pm_kisan_content),
        ("pmfby_crop_insurance_overview.txt", pmfby_content)
    ]
    for filename, content in scheme_files:
        filepath = os.path.join(RAW_DATA_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content.strip() + "\n")
        print(f"Saved Scheme document: {filepath}")


def main():
    print("--- Starting Data Download and Generation Pipeline ---")
    # 1. Download/create sample KCC dataset
    download_sample_kcc_data(RAW_DATA_DIR)
    
    # 2. Public agricultural advisory text files
    download_public_advisories()

    # 3. ICAR-style technical text files
    create_icar_advisories()

    # 4. Government scheme text files
    create_government_schemes()

    print("--- Completed Data Preparation Successfully ---")


if __name__ == "__main__":
    main()
