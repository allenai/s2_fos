"""
Write integration tests for your model interface code here.

The TestCase class below is supplied a `container`
to each test method. This `container` object is a proxy to the
Dockerized application running your model. It exposes a single method:

```
predict_batch(instances: List[Instance]) -> List[Prediction]
```

To test your code, create `Instance`s and make normal `TestCase`
assertions against the returned `Prediction`s.

e.g.

```
def test_prediction(self, container):
    instances = [Instance(), Instance()]
    predictions = container.predict_batch(instances)

    self.assertEqual(len(instances), len(predictions)

    self.assertEqual(predictions[0].field1, "asdf")
    self.assertGreatEqual(predictions[1].field2, 2.0)
```
"""


import logging
import sys
import unittest

from .interface import Instance, DecisionScore, Prediction


try:
    from timo_interface import with_timo_container
except ImportError as e:
    logging.warning(
        """
    This test can only be run by a TIMO test runner. No tests will run. 
    You may need to add this file to your project's pytest exclusions.
    """
    )
    sys.exit(0)


TEST_DATA = [
    {"title": "Ultrarotes Reflexionsverm√∂gen von SiO2"},
    {"title": "The physical geography of Fennoscandia"},
    {"title": "Comments: HIPAA Confusion: How the Privacy Rule Authorizes " '"Informal" Discovery'},
    {
        "title": "Imagining Spectatorship: From the Mysteries to the Shakespearean "
        "Stage. John J. McGavin and Greg Walker. Oxford Textual "
        "Perspectives. Oxford: Oxford University Press, 2016. xii + 208 pp. "
        "$29.95."
    },
    {"title": "\n Biological Psychiatry: Increasing Impact"},
    {
        "abstract": "Hierarchical structure porous MoS2/SiO2 microspheres were "
        "prepared by ultrasonic pyrolysis technique. The nanostructured "
        "MoS2/SiO2 materials were characterised by scanning electron "
        "micrograph (SEM), energy dispersive X-ray spectroscopy (EDX), "
        "X-ray diffraction (XRD), high-resolution transmission electron "
        "microscope (HRTEM), as well as nitrogen isotherm. The MoS2/SiO2 "
        "microspheres, synthesised using polystyrene latex spheres as a "
        "template, showed two pore sizes: 5.8 and 68\u2009nm. The "
        "micro-, meso- and macropore volume was also calculated. Effect "
        "of PSL:SiO2 ratio on the hierarchical structure was also "
        "investigated. ¬© 2011 Canadian Society for Chemical Engineering",
        "title": "Synthesis of hierarchical structured porous MoS2/SiO2 microspheres by ultrasonic spray pyrolysis",
    },
    {
        "abstract": "Abstract Copulas are fast gaining in popularity in fields "
        "requiring modelling of multivariate data. Population synthesis "
        "is one such domain where copulas hold much promise. "
        "Characteristics of a population are inherently a multivariate "
        "distribution, for example, age, education and employment have a "
        "dependent relationship with each other. However, "
        "characteristics of population are often discrete, sometimes "
        "even categorical, and rarely continuous. Further, inherently "
        "continuous variables like age are often discretized into bins "
        "or rounded down to the nearest integer. Although the Sklar‚Äôs "
        "representation theorem of copulas [23] is still valid for these "
        "discrete data, assessing the goodness of fit by rejecting the "
        "null hypothesis that an assumed family of copulas adequately "
        "represents the data may not be fool-proof. In this paper, "
        "through simulation we demonstrate the failure of accepted "
        "goodness of fit tests of copulas even when the copula family is "
        "able to capture the dependence in the data. The strong "
        "recommendation is to use additional methods to test that the "
        "copula can capture the dependence in the available data.",
        "title": "On Modelling Human Population Characteristics with Copulas",
    },
    {
        "abstract": "The Global Position System(GPS) is widely used for its great "
        "performance.But it utilizes low power spread-spectrum signals "
        "and thus is vulnerable to various types of strong interference "
        "sources.The smart antenna technology was used for suppressing "
        "jammers to GPS receivers.According to the features of GPS "
        "signal,a novel Noise RLS algorithm was proposed.The noise "
        "signal was introduced into RLS algorithm as reference signal to "
        "improve the algorithm.The simulation results show that,using "
        "N-RLS algorithm,higher signal-to-interference ratio can be "
        "obtained without reducing the degrees of freedom.",
        "title": "Novel Anti-Jamming Adaptive Beamforming Algorithm for GPS",
    },
    {
        "abstract": "The Las Vegas Valley Water District in Nevada, USA, has "
        "operated an artificial recharge (AR) program since 1989. In "
        "summer 2001, observations of gas exsolving from tap water "
        "prompted a study that revealed total dissolved gas (TDG) "
        "pressures approaching 2\xa0atm with a gas composition that it "
        "is predominantly air. Measurements of TDG pressure at well "
        "heads and in the distribution system indicated two potential "
        "mechanisms for elevated TDG pressures: (1) air entrainment "
        "during AR operations, and (2) temperature changes between the "
        "winter recharge season and the summer withdrawal season. Air "
        "entrainment during pumping was investigated by intentionally "
        "allowing the forebay (upstream reservoir) of a large pumping "
        "station to drawdown to the point of vortex formation. This "
        "resulted in up to a 0.7\xa0atm increase in TDG pressure. In "
        "general, the solubility of gases in water decreases as the "
        "temperature increases. In the Las Vegas Valley, water that "
        "acquired a modest amount of dissolved gas during winter "
        "artificial recharge operations experienced an increase in "
        "dissolved gas pressure (0.04\xa0atm/¬įC) as the water warmed in "
        "the subsurface. A combination of air entrainment during AR "
        "operations and its amplification by temperature increase after "
        "recharge can account for most of the observed amounts of excess "
        "gas at this site.R√©sum√©Le Las Vegas Valley Water District, "
        "Nevada, USA, a conduit un programme de recharge artificielle "
        "(AR) depuis 1989. En √©t√© 2001, des d√©gagements gazeux dans "
        "l‚Äôeau de conduite ont d√©clench√© une √©tude qui a r√©v√©l√© des "
        "pressions de gaz totaux dissous (TDG) approchant 2\xa0atm, "
        "constitu√©s principalement d‚Äôair. Des mesures de pression TDG en "
        "t√™te de puits et dans le r√©seau de distribution indiquaient "
        "deux m√©canismes potentiels expliquant des pressions TDG "
        "√©lev√©es: (1) entrainement d‚Äôair pendant les op√©rations de "
        "recharge (AR), et (2) changements de temp√©rature entre la "
        "saison de recharge et la saison de pr√©l√®vement en √©t√©. "
        "L‚Äôentrainement d‚Äôair durant le pr√©l√®vement a √©t√© √©tudi√© en "
        "pemettant intentionnellement √† la b√Ęche (le r√©servoir amont) "
        "d‚Äôune grande station de pompage de rabattre jusqu‚Äôau point de "
        "formation du vortex. Il en a r√©sult√© une augmentation de "
        "pression des gaz dissous (TDG) jusqu‚Äô√† 0.7\xa0atm. En g√©n√©ral, "
        "la solubilit√© des gaz dans l‚Äôeau d√©cro√ģt avec l‚Äôaugmentation de "
        "temp√©rature. Dans la vall√©e de Las Vegas, l‚Äôeau qui avait "
        "dissous un volume de gaz modeste durant la recharge "
        "artificielle hivernale montrait une augmentation de la presion "
        "de gaz dissous (0.04\xa0atm/¬įC) lorsque l‚Äôeau se r√©chauffait en "
        "sub-surface. La combinaison d‚Äôentrainement d‚Äôair durant la "
        "recharge (AR) et sa dilatation par √©l√©vation de la temp√©rature "
        "apr√®s recharge peut expliquer la plupart des exc√®s de gaz sur "
        "ce site.ResumenEl Las Vegas Valley Water District en Nevada, "
        "EEUU, ha operado una programa de recarga artificial (AR) desde "
        "1989. En el verano de 2001, las observaciones del gas "
        "proveniente de grifos de agua impuls√≥ un estudio que revel√≥ que "
        "la presi√≥n del gas disuelto total (TDG) que se aproximaba 2\xa0"
        "atm con una composici√≥n del gas es predominantemente aire. Las "
        "mediciones de la presi√≥n de TDG en las cargas de los pozos y en "
        "el sistema de distribuci√≥n indicaron dos mecanismos potenciales "
        "para las elevadas presiones de TDG: (1) captura del aire "
        "durante las operaciones de AR, y (2) cambios en las "
        "temperaturas entre recarga invernsal y la extracci√≥n estival. "
        "La captura del aire durante el bombeo fue investigado "
        "permitiendo intencionalmente el movimiento (aguas arriba del "
        "embalse) de una gran estaci√≥n de bombeo para la extracci√≥n "
        "hasta el punto de formaci√≥n del v√≥rtice. Esto result√≥ en hasta "
        "un incremento de 0.7\xa0atm en la presi√≥n de TDG. En general, "
        "la solubilidad de los gases en el agua decrece a medida que se "
        "incrementa la temperatura. En Las Vegas Valley, el agua que "
        "adquiri√≥ una cantidad modesta de gases disueltos durante las "
        "operaciones de recarga artificial invernal experiment√≥ un "
        "incremento en la presi√≥n del gas disuelto (0.04\xa0atm/ňöC) a "
        "medida que el agua se calentaba en la subsuperficie. Una "
        "combinaci√≥n del aire capturado durante las operaciones de AR y "
        "su amplificaci√≥n por el incremento de la temperatura despu√©s de "
        "la recarga pueden explicar la mayor parte de las cantidades "
        "excesivas observadas de gas en el "
        "sitio.śĎėŤ¶ĀÁĺéŚõĹŚÜÖŚćéŤĺĺŚ∑ěśčČśĖĮÁĽīŚä†śĖĮŚ≥°Ťį∑śįīŚĆļŤá™1989ŚĻīšĽ•śĚ•, ŚģěśĖĹšļÜšłÄšł™šļļŚ∑•Ť°•ÁĽôÁöĄť°ĻÁõģ„ÄāŚú®2001ŚĻīŚ§ŹŚ§©, "
        "Ťá™śĚ•śįīšł≠ÁöĄśįĒšĹďŤĄĪśļ∂ÁéįŤĪ°šŅÉŤŅõšļÜšłÄť°ĻÁ†ĒÁ©∂, ŤĮ•Á†ĒÁ©∂śŹ≠Á§ļšļÜśÄĽśļ∂Ťß£śįĒšĹď (TDG) "
        "ŚéčŚäõŤĺĺŚąįšł§šł™Ś§ßśįĒŚéčšłĒśįĒšĹďšłĽŤ¶ĀśąźŚąÜśėĮÁ©ļśįĒ„ÄāšļēŚŹ£ŚíĆťÖćśįīÁ≥ĽÁĽüšł≠TDGŚéčŚäõÁöĄśĶčťáŹŤ°®śėéśŹźťęėÁöĄśÄĽśļ∂Ťß£śįĒšĹďŚéčŚäõśúČšł§ÁßćŚŹĮŤÉĹÁöĄśúļŚą∂: (1) "
        "šļļŚ∑•Ť°•ÁĽôŤŅáÁ®čšł≠ÁöĄÁ©ļśįĒŤĺďťÄĀ, ŚíĆ (2) ŚÜ¨Ś§©Ť°•ÁĽôŚ≠£ŤäāŚíĆŚ§ŹŚ§©ŚľÄťááŚ≠£ŤäāšĻčťóīÁöĄśł©Śļ¶ŚŹėŚĆĖ„ÄāťÄöŤŅáśúČśĄŹŚúįšĹŅšłäśłł (šłäśłłśįīŚļď) "
        "ÁöĄŚ§ßŚěčÁöĄśäĹśįīÁęôÁöĄśįīšĹćšłčťôćŤá≥ŤÉĹŚĹĘśąźśľ©ś∂°ÁöĄÁāĻ, "
        "ŚĮĻśäĹśįīŤŅáÁ®čšł≠ÁöĄÁ©ļśįĒŤĺďŚÖ•ŤŅõŤ°ĆšļÜÁ†ĒÁ©∂„ÄāŤŅôŚįĪŚĮľŤáīśÄĽśļ∂Ťß£śįĒšĹďŚéčŚäõśúČŤá≥Ś§ö0.7Ś§ßśįĒŚéčÁöĄŚĘěŚä†„ÄāśÄĽÁöĄśĚ•ŤĮī, "
        "śįīšł≠śįĒšĹďśļ∂Ťß£Śļ¶ťöŹśł©Śļ¶ÁöĄŚĘěŚä†ŤÄĆťôćšĹé„ÄāŚú®śčČśĖĮÁĽīŚä†śĖĮŚ≥°Ťį∑, "
        "ŚÜ¨Ś≠£šļļŚ∑•Ť°•ÁĽôśé™śĖĹŤŅáÁ®čšł≠śļ∂Ťß£šļÜÁõłŚĹďťáŹśįĒšĹďÁöĄśįīšł≠ÁöĄśļ∂Ťß£śįĒŚéčŚäõťöŹÁĚÄśįīŚú®ŚúįšłčŚŹėśöĖŤÄĆŚĘěŚä†šļÜ (0.04\xa0atm/ňöC) "
        "„ÄāšļļŚ∑•Ť°•ÁĽôŤŅáÁ®čšł≠ÁöĄÁ©ļśįĒŤĺďŚÖ•ŚíĆŤ°•ÁĽôŚźéśł©Śļ¶ŚćáťęėŚĮľŤáīÁöĄŚĘěŚä†śėĮŤŅôšł™ŚúįŚĆļŚ≠ėŚú®śįĒšĹďŤŅáŚČ©ÁöĄšłĽŤ¶ĀŚéüŚõ†„ÄāResumoA "
        "administra√ß√£o hidr√°ulica de Las Vegas Valley, Nevada, EUA, "
        "opera um programa de recarga artificial (RA) desde 1989. No "
        "ver√£o de 2001, a observa√ß√£o da exsolu√ß√£o de g√°s nos pontos de "
        "consumo desencadeou um estudo que revelou press√Ķes de g√°s "
        "dissolvido total (GDT) pr√≥ximas de 2\xa0atm com uma composi√ß√£o "
        "gasosa predominante de ar. As medidas de press√£o de GDT √† "
        "cabe√ßa dos po√ßos e no sistema de distribui√ß√£o apontaram para "
        "dois mecanismos potenciais para as elevadas press√Ķes de GDT: "
        "(1) arrastamento de ar durante as opera√ß√Ķes de RA e (2) "
        "varia√ß√Ķes de temperatura entre a √©poca de recarga invernal e a "
        "√©poca de extrac√ß√£o no ver√£o. Foi investigado o arrastamento de "
        "ar durante a bombagem permitindo intencionalmente que a bacia "
        "de tomada de √°gua de uma grande central de bombagem rebaixasse "
        "at√© ao ponto de forma√ß√£o de v√≥rtex. Isto resultou num aumento "
        "de at√© 0.7\xa0atm na press√£o do GDT. Em geral, a solubilidade "
        "dos gases na √°gua diminui com o aumento da temperatura. Em Las "
        "Vegas Valley, a √°gua, que adquiriu uma reduzida quantidade de "
        "gases dissolvidos durante as opera√ß√Ķes de recarga no inverno, "
        "sofreu um aumento da press√£o dos gases dissolvidos (0.04\xa0"
        "atm/ňöC) quando aqueceu no meio subterr√Ęneo. A combina√ß√£o do "
        "arrastamento de ar durante as opera√ß√Ķes de RA e da sua "
        "amplia√ß√£o devida ao incremento de temperatura depois da recarga "
        "podem explicar a maioria das quantidades de g√°s excessivo "
        "observadas neste local.",
        "title": "Excess air during aquifer storage and recovery in an arid basin " "(Las Vegas Valley, USA)",
    },
    {
        "abstract": "ABSTRACT This study was undertaken to validate the ‚Äúquick, "
        "easy, cheap, effective, rugged and safe‚ÄĚ (QuEChERS) method "
        "using Golden Delicious and Starking Delicious apple matrices "
        "spiked at 0.1 maximum residue limit (MRL), 1.0 MRL and 10 MRL "
        "levels of the four pesticides (chlorpyrifos, dimethoate, "
        "indoxacarb and imidacloprid). For the extraction and cleanup, "
        "original QuEChERS method was followed, then the samples were "
        "subjected to liquid chromatography-triple quadrupole mass "
        "spectrometry (LC-MS/MS) for chromatographic analyses. According "
        "to t test, matrix effect was not significant for chlorpyrifos "
        "in both sample matrices, but it was significant for dimethoate, "
        "indoxacarb and imidacloprid in both sample matrices. Thus, "
        "matrix-matched calibration (MC) was used to compensate matrix "
        "effect and quantifications were carried out by using MC. The "
        "overall recovery of the method was 90.15% with a relative "
        "standard deviation of 13.27% (n = 330). Estimated method "
        "detection limit of analytes blew the MRLs. Some other "
        "parameters of the method validation, such as recovery, "
        "precision, accuracy and linearity were found to be within the "
        "required ranges.",
        "title": "Validation of QuEChERS method for the determination of some "
        "pesticide residues in two apple varieties",
    },
]

TEST_EXPECTED_PREDICTIONS = [
    [],
    [
        {"label": "Agricultural and Food sciences", "score": -1.1361256722621955},
        {"label": "Art", "score": -1.199113417021466},
        {"label": "Biology", "score": -1.305630086094644},
        {"label": "Business", "score": -1.9235805898455751},
        {"label": "Chemistry", "score": -1.7077950419751773},
        {"label": "Computer science", "score": -1.4921318857892263},
        {"label": "Economics", "score": -0.38839550587512306},
        {"label": "Education", "score": -1.5322326730009055},
        {"label": "Engineering", "score": -1.23565093889931},
        {"label": "Environmental science", "score": -0.5841672745909818},
        {"label": "Geography", "score": -0.6020386607143793},
        {"label": "Geology", "score": -1.159910176232368},
        {"label": "History", "score": -0.8466238106885753},
        {"label": "Law", "score": -1.4489646926061843},
        {"label": "Linguistics", "score": -1.6488616358657258},
        {"label": "Materials science", "score": -1.3179397306188558},
        {"label": "Mathematics", "score": -1.032266518298589},
        {"label": "Medicine", "score": -0.9755598587875712},
        {"label": "Philosophy", "score": -1.2259603901157163},
        {"label": "Physics", "score": -1.1852970904427997},
        {"label": "Political science", "score": -1.576997378386134},
        {"label": "Psychology", "score": -1.8692432447555347},
        {"label": "Sociology", "score": -1.1426934998166784},
    ],
    [
        {"label": "Agricultural and Food sciences", "score": -1.4544872657142176},
        {"label": "Art", "score": -0.626223131265735},
        {"label": "Biology", "score": -1.6198467626367719},
        {"label": "Business", "score": -1.1395157832176879},
        {"label": "Chemistry", "score": -1.6331084304618388},
        {"label": "Computer science", "score": -0.2828423096194519},
        {"label": "Economics", "score": -0.5137075708069005},
        {"label": "Education", "score": -0.8346982920491277},
        {"label": "Engineering", "score": -1.398572081959636},
        {"label": "Environmental science", "score": -1.5763889083268192},
        {"label": "Geography", "score": -1.7282669669268445},
        {"label": "Geology", "score": -1.7467824373264345},
        {"label": "History", "score": -1.0923292849275947},
        {"label": "Law", "score": 0.12047282546604712},
        {"label": "Linguistics", "score": -1.5010324972066855},
        {"label": "Materials science", "score": -1.7060667455609042},
        {"label": "Mathematics", "score": -1.152732215069817},
        {"label": "Medicine", "score": -0.7350130410230582},
        {"label": "Philosophy", "score": -1.2796939242223115},
        {"label": "Physics", "score": -0.9341335112889777},
        {"label": "Political science", "score": -1.211318676558134},
        {"label": "Psychology", "score": -0.8820761945261552},
        {"label": "Sociology", "score": -1.485643735918854},
    ],
    [
        {"label": "Agricultural and Food sciences", "score": -1.4239843282967823},
        {"label": "Art", "score": -1.025892982937838},
        {"label": "Biology", "score": -1.094041166209622},
        {"label": "Business", "score": -1.7382875934286823},
        {"label": "Chemistry", "score": -1.362001947496357},
        {"label": "Computer science", "score": -1.7213482571318517},
        {"label": "Economics", "score": -1.8008458046138318},
        {"label": "Education", "score": -1.2555653334584147},
        {"label": "Engineering", "score": -1.6790278734388346},
        {"label": "Environmental science", "score": -1.1719532029475377},
        {"label": "Geography", "score": -1.3773785785078871},
        {"label": "Geology", "score": -1.7050145450865433},
        {"label": "History", "score": -1.1760048459887051},
        {"label": "Law", "score": -1.745381331151223},
        {"label": "Linguistics", "score": -0.14206199970499836},
        {"label": "Materials science", "score": -1.2635959699325272},
        {"label": "Mathematics", "score": -1.325303305674634},
        {"label": "Medicine", "score": -1.45720914394753},
        {"label": "Philosophy", "score": -0.7952447835084245},
        {"label": "Physics", "score": -1.5414491448946144},
        {"label": "Political science", "score": -1.016667506388898},
        {"label": "Psychology", "score": -0.7989576376037708},
        {"label": "Sociology", "score": -0.9438802973878054},
    ],
    [
        {"label": "Agricultural and Food sciences", "score": -1.5402338179249464},
        {"label": "Art", "score": -1.8466683968200195},
        {"label": "Biology", "score": -1.1872768295595246},
        {"label": "Business", "score": -1.1751807659774927},
        {"label": "Chemistry", "score": -1.7186691762966027},
        {"label": "Computer science", "score": -1.8677371427100946},
        {"label": "Economics", "score": -1.9585465812840446},
        {"label": "Education", "score": -1.4440402589169037},
        {"label": "Engineering", "score": -1.395376114620939},
        {"label": "Environmental science", "score": -1.7028355863385576},
        {"label": "Geography", "score": -1.3952594935504627},
        {"label": "Geology", "score": -2.145565725566887},
        {"label": "History", "score": -3.2666700376181224},
        {"label": "Law", "score": -1.571181898246024},
        {"label": "Linguistics", "score": -2.1026294252409854},
        {"label": "Materials science", "score": -1.6842703420011649},
        {"label": "Mathematics", "score": -1.9224378691471724},
        {"label": "Medicine", "score": 0.3652632849122995},
        {"label": "Philosophy", "score": -1.7157507887983061},
        {"label": "Physics", "score": -1.9387966339417704},
        {"label": "Political science", "score": -2.057189670678518},
        {"label": "Psychology", "score": 1.072940423723258},
        {"label": "Sociology", "score": -1.6187922370991563},
    ],
    [
        {"label": "Agricultural and Food sciences", "score": -1.3102171018622062},
        {"label": "Art", "score": -2.2134269624910963},
        {"label": "Biology", "score": -1.6767717914030926},
        {"label": "Business", "score": -2.313053647198438},
        {"label": "Chemistry", "score": -0.2888393742144273},
        {"label": "Computer science", "score": -1.5852683679140043},
        {"label": "Economics", "score": -2.7260141254150763},
        {"label": "Education", "score": -1.981892530162807},
        {"label": "Engineering", "score": -0.6345205672488098},
        {"label": "Environmental science", "score": -1.427039115533614},
        {"label": "Geography", "score": -1.5716832709621478},
        {"label": "Geology", "score": -1.5599972405463287},
        {"label": "History", "score": -2.3123378381613},
        {"label": "Law", "score": -2.2790072690637357},
        {"label": "Linguistics", "score": -1.9150393041617504},
        {"label": "Materials science", "score": 0.8967058158854587},
        {"label": "Mathematics", "score": -1.9977084464688286},
        {"label": "Medicine", "score": -1.3086587854849385},
        {"label": "Philosophy", "score": -2.3849110954284103},
        {"label": "Physics", "score": -0.6949242840262054},
        {"label": "Political science", "score": -1.9387768215518875},
        {"label": "Psychology", "score": -2.00983002399056},
        {"label": "Sociology", "score": -2.094257944148292},
    ],
    [
        {"label": "Agricultural and Food sciences", "score": -1.9014446151758346},
        {"label": "Art", "score": -1.9464084409349174},
        {"label": "Biology", "score": -1.0030439450651658},
        {"label": "Business", "score": -1.3723654382648718},
        {"label": "Chemistry", "score": -1.4985545259338573},
        {"label": "Computer science", "score": 0.5638193790888248},
        {"label": "Economics", "score": -0.7785092899904701},
        {"label": "Education", "score": -0.9208310995186084},
        {"label": "Engineering", "score": -0.6809169426299904},
        {"label": "Environmental science", "score": -1.1808801010876895},
        {"label": "Geography", "score": -1.0221694012125213},
        {"label": "Geology", "score": -1.524179041900549},
        {"label": "History", "score": -1.4451031016746705},
        {"label": "Law", "score": -1.4234133070270647},
        {"label": "Linguistics", "score": -1.3100696975526058},
        {"label": "Materials science", "score": -1.6881123444273802},
        {"label": "Mathematics", "score": -0.8087380663623378},
        {"label": "Medicine", "score": -1.2297353117935403},
        {"label": "Philosophy", "score": -1.5956293054578599},
        {"label": "Physics", "score": -1.437607409970327},
        {"label": "Political science", "score": -0.9695106742019584},
        {"label": "Psychology", "score": -1.064602055055281},
        {"label": "Sociology", "score": -1.1863719575994687},
    ],
    [
        {"label": "Agricultural and Food sciences", "score": -1.6750465689227767},
        {"label": "Art", "score": -1.6933520356524603},
        {"label": "Biology", "score": -1.836707542889642},
        {"label": "Business", "score": -0.8060187763532238},
        {"label": "Chemistry", "score": -1.266583169437207},
        {"label": "Computer science", "score": -0.3055044307430998},
        {"label": "Economics", "score": -1.962798873149338},
        {"label": "Education", "score": -1.9593376316335966},
        {"label": "Engineering", "score": -0.0737538388680587},
        {"label": "Environmental science", "score": -1.5010775405244001},
        {"label": "Geography", "score": -2.0393705329923266},
        {"label": "Geology", "score": -1.42019666153939},
        {"label": "History", "score": -2.2820865846515894},
        {"label": "Law", "score": -2.2071626717737782},
        {"label": "Linguistics", "score": -2.166499201968539},
        {"label": "Materials science", "score": -0.9372081113265532},
        {"label": "Mathematics", "score": -2.033156353335323},
        {"label": "Medicine", "score": -0.8788416402355527},
        {"label": "Philosophy", "score": -1.8724182716916915},
        {"label": "Physics", "score": -1.2326826885592064},
        {"label": "Political science", "score": -1.4231845278004314},
        {"label": "Psychology", "score": -1.656086301674007},
        {"label": "Sociology", "score": -2.308352013994598},
    ],
    [],
    [
        {"label": "Agricultural and Food sciences", "score": -0.09823857333606328},
        {"label": "Art", "score": -2.5382556613504654},
        {"label": "Biology", "score": -1.0199257204277847},
        {"label": "Business", "score": -1.6148420712406624},
        {"label": "Chemistry", "score": -0.4236274626696126},
        {"label": "Computer science", "score": -1.2066650125834137},
        {"label": "Economics", "score": -1.6869219343685586},
        {"label": "Education", "score": -1.725973506208393},
        {"label": "Engineering", "score": -1.2287175219255726},
        {"label": "Environmental science", "score": -1.3583292749845182},
        {"label": "Geography", "score": -1.538220001590919},
        {"label": "Geology", "score": -2.0471031233883936},
        {"label": "History", "score": -2.1693398794605723},
        {"label": "Law", "score": -1.9080929723557143},
        {"label": "Linguistics", "score": -1.7882896678712183},
        {"label": "Materials science", "score": -1.5263378351652432},
        {"label": "Mathematics", "score": -1.4865795596777283},
        {"label": "Medicine", "score": -1.5888382709471738},
        {"label": "Philosophy", "score": -1.8618186715883662},
        {"label": "Physics", "score": -1.347066212988397},
        {"label": "Political science", "score": -1.8736253356675792},
        {"label": "Psychology", "score": -1.5239175462570778},
        {"label": "Sociology", "score": -1.678497971185068},
    ],
]


@with_timo_container
class TestInterfaceIntegration(unittest.TestCase):
    def test__predictions(self, container):

        expected_predictions = []
        for pred in TEST_EXPECTED_PREDICTIONS:
            expected_predictions.append(Prediction(scores=[DecisionScore(**i) for i in pred]))

        instances = []
        for test_datum in TEST_DATA:
            instances.append(Instance(**test_datum))

        predictions = container.predict_batch(instances)

        for pred_expected, pred_made in zip(expected_predictions, predictions):
            self.assertEqual(pred_expected, pred_made)
