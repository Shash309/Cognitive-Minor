import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';

const resources = {
  en: {
    translation: {
      common: {
        appName: 'Career Explorer',
        insights: 'Insights',
        welcomeUser: 'Welcome, {{name}}!',
        logout: 'Logout',
        dashboard: 'Dashboard',
        exploreColleges: 'Explore Colleges',
        aiCareerQuiz: 'AI Career Quiz',
        skillBuilder: 'Skill Builder',
        careerVisualizer: 'Career Visualizer',
        timelineTracker: 'Timeline Tracker',
        retry: 'Retry',
        learnMore: 'Learn More',
        visitWebsite: 'Visit Website',
        viewDetails: 'View Details',
      },
      landing: {
        createAccount: 'Create Account',
        orUseEmail: 'or use your email for registration',
        name: 'Name',
        email: 'Email',
        password: 'Password',
        signUp: 'Sign Up',
        signIn: 'Sign In',
        signInTitle: 'Sign in',
        orUseAccount: 'or use your account',
        forgotPassword: 'Forgot your password?',
        welcomeBack: 'Welcome Back!',
        keepConnected: 'To keep connected with us please login with your personal info',
        helloFriend: 'Hello, Friend!',
        startJourney: 'Enter your personal details and start your journey with us'
      },
      dashboard: {
        exploreCollegesDesc: 'Search and filter colleges across India by state and rank.',
        aiCareerQuizDesc: 'Answer questions to get a personalized career recommendation.',
        skillBuilderDesc: 'Discover the key skills required for your chosen career path.',
        careerVisualizerDesc: 'Visually explore the connections between subjects, degrees, and careers.',
        timelineTrackerDesc: 'Stay updated on all important admission and scholarship dates.',
        welcomeBackUser: 'Welcome back, {{name}}!',
        welcomeSub: 'Your journey to a successful career continues here. What would you like to do today?'
      },
      quiz: {
        yourRecommendationTitle: 'Your Personalized Recommendation',
        recommendedPath: 'Recommended Path:',
        retakeQuiz: 'Retake Quiz',
        back: 'Back',
        next: 'Next',
        submit: 'Get Results ✨',
        predicting: 'Predicting...',
        questions: {
          Q1: { text: 'What are your favorite subjects?' },
          Q2: { text: 'Which activities do you enjoy most?' },
          Q3: { text: 'What do you consider your strongest skills?' },
          Q4: { text: 'Which work style suits you better?' },
          Q5: { text: 'What type of workplace do you prefer?' },
          Q6: { text: 'Are you ready for competitive exams?' },
          Q7: { text: 'Where would you prefer to study/work?' },
          Q8: { text: 'What career values matter most to you?' },
          Q9: { text: 'What is your long-term career goal?' },
          Q10: { text: 'What is your academic background (with %)?' }
        }
      },
      college: {
        title: 'Find Colleges by State',
        selectStatePlaceholder: '-- Select a State --',
        featured: 'Featured',
        rank: 'Rank',
        score: 'Score',
        errorStates: 'Failed to fetch states. Please check backend.',
        errorCollegesPrefix: 'Failed to fetch colleges: ',
        tier1InState: 'Tier-1 Colleges in {{state}}',
        tier2InState: 'Tier-2 Colleges in {{state}}',
        othersInState: 'Other Notable Colleges in {{state}}',
        noCollegesForState: 'No colleges found for {{state}}.',
        promptSelectState: 'Please select a state from the dropdown to begin.'
      },
      visualizer: {
        title: 'Dynamic Career Path Visualizer',
        subtitle: 'Hover over a node to highlight its path. Click to learn more!',
        legendSubjects: 'Subjects',
        legendDegrees: 'Degrees',
        legendCareers: 'Careers'
      },
      skills: {
        title: 'Skill Builder – Learn What Matters',
        subtitle: 'Here are some of the most in-demand skills for top careers. Click to learn more.',
        items: {
          problemSolving: { name: 'Problem Solving', description: 'Essential for engineering and tech roles.' },
          communication: { name: 'Communication', description: 'Key for teamwork and leadership.' },
          dsa: { name: 'Data Structures & Algorithms', description: 'The foundation of computer science.' },
          projectManagement: { name: 'Project Management', description: 'Learn to plan and execute projects efficiently.' },
          cloud: { name: 'Cloud Computing (AWS/Azure)', description: 'High-demand skill in the modern tech landscape.' }
        }
      },
      timeline: {
        title: 'Admissions & Scholarship Timeline',
        subtitle: 'Stay updated with all the important upcoming deadlines for exams, admissions, and scholarships.',
        filters: {
          all: 'All',
          exams: 'Exams',
          scholarships: 'Scholarships',
          admissions: 'Admissions'
        },
        searchPlaceholder: 'Search for an event...',
        errorFetch: 'Could not fetch events. Please try again later.',
        status: {
          closed: 'Closed',
          active: 'Active',
          upcoming: 'Upcoming'
        },
        deadlineLabel: 'Deadline:',
        noEvents: 'No events found for your current search or filter.'
      }
    }
  },
  hi: {
    translation: {
      common: {
        appName: 'कैरियर एक्सप्लोरर',
        insights: 'इनसाइट्स',
        welcomeUser: 'स्वागत है, {{name}}!',
        logout: 'लॉग आउट',
        dashboard: 'डैशबोर्ड',
        exploreColleges: 'कॉलेज खोजें',
        aiCareerQuiz: 'एआई करियर क्विज़',
        skillBuilder: 'स्किल बिल्डर',
        careerVisualizer: 'करियर विज़ुअलाइज़र',
        timelineTracker: 'टाइमलाइन ट्रैकर',
        retry: 'पुनः प्रयास करें',
        learnMore: 'और जानें',
        visitWebsite: 'वेबसाइट देखें',
        viewDetails: 'विवरण देखें',
      },
      landing: {
        createAccount: 'खाता बनाएं',
        orUseEmail: 'या पंजीकरण के लिए अपना ईमेल उपयोग करें',
        name: 'नाम',
        email: 'ईमेल',
        password: 'पासवर्ड',
        signUp: 'साइन अप',
        signIn: 'साइन इन',
        signInTitle: 'साइन इन',
        orUseAccount: 'या अपना खाता उपयोग करें',
        forgotPassword: 'क्या आप अपना पासवर्ड भूल गए?',
        welcomeBack: 'वापसी पर स्वागत है!',
        keepConnected: 'जुड़े रहने के लिए कृपया अपनी व्यक्तिगत जानकारी से लॉगिन करें',
        helloFriend: 'नमस्ते, मित्र!',
        startJourney: 'अपनी व्यक्तिगत जानकारी दर्ज करें और हमारी यात्रा शुरू करें'
      },
      dashboard: {
        exploreCollegesDesc: 'राज्य और रैंक के आधार पर भारत भर के कॉलेज खोजें।',
        aiCareerQuizDesc: 'निजीकृत करियर अनुशंसा पाने के लिए प्रश्नों के उत्तर दें।',
        skillBuilderDesc: 'अपने चुने हुए करियर पथ के लिए आवश्यक प्रमुख कौशल खोजें।',
        careerVisualizerDesc: 'विषयों, डिग्री और करियर के संबंधों को दृश्य रूप से देखें।',
        timelineTrackerDesc: 'महत्वपूर्ण प्रवेश और छात्रवृत्ति तिथियों से अपडेट रहें।',
        welcomeBackUser: 'वापसी पर स्वागत है, {{name}}!',
        welcomeSub: 'सफल करियर की आपकी यात्रा यहाँ जारी है। आज आप क्या करना चाहेंगे?'
      },
      quiz: {
        yourRecommendationTitle: 'आपकी व्यक्तिगत अनुशंसा',
        recommendedPath: 'सुझाया गया पथ:',
        retakeQuiz: 'क्विज़ फिर से लें',
        back: 'पीछे',
        next: 'आगे',
        submit: 'परिणाम प्राप्त करें ✨',
        predicting: 'अनुमान लगाया जा रहा है...',
        questions: {
          Q1: { text: 'आपके पसंदीदा विषय कौन से हैं?' },
          Q2: { text: 'आप किन गतिविधियों का सबसे अधिक आनंद लेते हैं?' },
          Q3: { text: 'आप किन कौशलों को अपनी सबसे बड़ी ताकत मानते हैं?' },
          Q4: { text: 'कौन-सी कार्यशैली आपको बेहतर लगती है?' },
          Q5: { text: 'आप किस प्रकार के कार्यस्थल को पसंद करते हैं?' },
          Q6: { text: 'क्या आप प्रतियोगी परीक्षाओं के लिए तैयार हैं?' },
          Q7: { text: 'आप कहाँ अध्ययन/काम करना पसंद करेंगे?' },
          Q8: { text: 'कौन-से करियर मूल्य आपके लिए सबसे महत्वपूर्ण हैं?' },
          Q9: { text: 'आपका दीर्घकालिक करियर लक्ष्य क्या है?' },
          Q10: { text: 'आपकी शैक्षणिक पृष्ठभूमि (प्रतिशत सहित) क्या है?' }
        }
      },
      college: {
        title: 'राज्य के अनुसार कॉलेज खोजें',
        selectStatePlaceholder: '-- कोई राज्य चुनें --',
        featured: 'विशेष',
        rank: 'रैंक',
        score: 'स्कोर',
        errorStates: 'राज्यों को प्राप्त करने में विफल। कृपया बैकएंड जाँचें।',
        errorCollegesPrefix: 'कॉलेज प्राप्त करने में विफल: ',
        tier1InState: '{{state}} में टियर-1 कॉलेज',
        tier2InState: '{{state}} में टियर-2 कॉलेज',
        othersInState: '{{state}} में अन्य उल्लेखनीय कॉलेज',
        noCollegesForState: '{{state}} के लिए कोई कॉलेज नहीं मिला।',
        promptSelectState: 'शुरू करने के लिए कृपया ड्रॉपडाउन से कोई राज्य चुनें।'
      },
      visualizer: {
        title: 'डायनेमिक करियर पाथ विज़ुअलाइज़र',
        subtitle: 'किसी नोड पर होवर करें पथ हाइलाइट करने के लिए। अधिक जानने के लिए क्लिक करें!',
        legendSubjects: 'विषय',
        legendDegrees: 'डिग्रियाँ',
        legendCareers: 'करियर'
      },
      skills: {
        title: 'स्किल बिल्डर – जरूरी बातें सीखें',
        subtitle: 'शीर्ष करियर के लिए सबसे अधिक मांग वाली कुछ कौशल यहाँ हैं। अधिक जानने के लिए क्लिक करें।',
        items: {
          problemSolving: { name: 'समस्या समाधान', description: 'इंजीनियरिंग और टेक भूमिकाओं के लिए आवश्यक।' },
          communication: { name: 'संचार', description: 'टीमवर्क और नेतृत्व के लिए महत्वपूर्ण।' },
          dsa: { name: 'डेटा स्ट्रक्चर्स और एल्गोरिद्म', description: 'कंप्यूटर विज्ञान की नींव।' },
          projectManagement: { name: 'प्रोजेक्ट प्रबंधन', description: 'प्रोजेक्ट्स की योजना और निष्पादन कुशलता से सीखें।' },
          cloud: { name: 'क्लाउड कंप्यूटिंग (AWS/Azure)', description: 'आधुनिक टेक परिदृश्य में उच्च मांग।' }
        }
      },
      timeline: {
        title: 'एडमिशन और स्कॉलरशिप टाइमलाइन',
        subtitle: 'परीक्षाओं, प्रवेश और छात्रवृत्ति की महत्वपूर्ण अंतिम तिथियों से अपडेट रहें।',
        filters: {
          all: 'सभी',
          exams: 'परीक्षाएँ',
          scholarships: 'स्कॉलरशिप',
          admissions: 'एडमिशन'
        },
        searchPlaceholder: 'कार्यक्रम खोजें...',
        errorFetch: 'ईवेंट्स प्राप्त नहीं कर सके। कृपया बाद में पुनः प्रयास करें।',
        status: {
          closed: 'समाप्त',
          active: 'सक्रिय',
          upcoming: 'आगामी'
        },
        deadlineLabel: 'अंतिम तिथि:',
        noEvents: 'आपकी वर्तमान खोज या फ़िल्टर के लिए कोई ईवेंट नहीं मिला।'
      }
    }
  },
  ur: {
    translation: {
      common: {
        appName: 'کیریئر ایکسپلورر',
        insights: 'اِن سائٹس',
        welcomeUser: 'خوش آمدید، {{name}}!',
        logout: 'لاگ آؤٹ',
        dashboard: 'ڈیش بورڈ',
        exploreColleges: 'کالجز تلاش کریں',
        aiCareerQuiz: 'اے آئی کیریئر کوئز',
        skillBuilder: 'اسکل بلڈر',
        careerVisualizer: 'کیریئر ویزؤلائزر',
        timelineTracker: 'ٹائم لائن ٹریکر',
        retry: 'دوبارہ کوشش کریں',
        learnMore: 'مزید جانیں',
        visitWebsite: 'ویب سائٹ دیکھیں',
        viewDetails: 'تفصیلات دیکھیں',
      },
      landing: {
        createAccount: 'اکاؤنٹ بنائیں',
        orUseEmail: 'یا رجسٹریشن کے لیے اپنا ای میل استعمال کریں',
        name: 'نام',
        email: 'ای میل',
        password: 'پاس ورڈ',
        signUp: 'سائن اپ',
        signIn: 'سائن اِن',
        signInTitle: 'سائن اِن',
        orUseAccount: 'یا اپنا اکاؤنٹ استعمال کریں',
        forgotPassword: 'کیا آپ اپنا پاس ورڈ بھول گئے؟',
        welcomeBack: 'خوش آمدید!',
        keepConnected: 'متصل رہنے کے لیے براہ کرم اپنی ذاتی معلومات سے لاگ اِن کریں',
        helloFriend: 'ہیلو، دوست!',
        startJourney: 'اپنی ذاتی معلومات درج کریں اور ہمارے ساتھ سفر شروع کریں'
      },
      dashboard: {
        exploreCollegesDesc: 'ریاست اور رینک کے مطابق پورے بھارت میں کالجز تلاش کریں۔',
        aiCareerQuizDesc: 'ذاتی نوعیت کی کیریئر تجویز کے لیے سوالات کے جواب دیں۔',
        skillBuilderDesc: 'اپنے منتخب کردہ کیریئر راستے کے لیے اہم مہارتیں معلوم کریں۔',
        careerVisualizerDesc: 'مضامین، ڈگریوں اور کیریئرز کے تعلقات کو بصری طور پر دیکھیں۔',
        timelineTrackerDesc: 'اہم داخلہ اور اسکالرشپ کی تاریخوں سے باخبر رہیں۔',
        welcomeBackUser: 'خوش آمدید، {{name}}!',
        welcomeSub: 'کامیاب کیریئر کی آپ کی سفر یہاں جاری ہے۔ آج آپ کیا کرنا چاہیں گے؟'
      },
      quiz: {
        yourRecommendationTitle: 'آپ کی ذاتی تجویز',
        recommendedPath: 'تجویز کردہ راستہ:',
        retakeQuiz: 'کوئز دوبارہ لیں',
        back: 'واپس',
        next: 'اگلا',
        submit: 'نتائج حاصل کریں ✨',
        predicting: 'اندازہ لگایا جا رہا ہے...',
        questions: {
          Q1: { text: 'آپ کے پسندیدہ مضامین کون سے ہیں؟' },
          Q2: { text: 'آپ کو کون سی سرگرمیاں سب سے زیادہ پسند ہیں؟' },
          Q3: { text: 'آپ کن مہارتوں کو اپنی سب سے بڑی طاقت سمجھتے ہیں؟' },
          Q4: { text: 'کون سی کام کرنے کی طرز آپ کے لیے بہتر ہے؟' },
          Q5: { text: 'آپ کس قسم کی جگہ پر کام/پڑھائی پسند کرتے ہیں؟' },
          Q6: { text: 'کیا آپ مقابلہ جاتی امتحانات کے لیے تیار ہیں؟' },
          Q7: { text: 'آپ کہاں پڑھنا/کام کرنا پسند کریں گے؟' },
          Q8: { text: 'آپ کے لیے کون سی کیریئر اقدار سب سے اہم ہیں؟' },
          Q9: { text: 'آپ کا طویل مدتی کیریئر ہدف کیا ہے؟' },
          Q10: { text: 'آپ کا تعلیمی پس منظر (فیصد کے ساتھ) کیا ہے؟' }
        }
      },
      college: {
        title: 'ریاست کے مطابق کالجز تلاش کریں',
        selectStatePlaceholder: '-- کوئی ریاست منتخب کریں --',
        featured: 'نمایاں',
        rank: 'رینک',
        score: 'اسکور',
        errorStates: 'ریاستیں حاصل کرنے میں ناکامی۔ براہ کرم بیک اینڈ چیک کریں۔',
        errorCollegesPrefix: 'کالجز حاصل کرنے میں ناکامی: ',
        tier1InState: '{{state}} میں ٹئیر-1 کالجز',
        tier2InState: '{{state}} میں ٹئیر-2 کالجز',
        othersInState: '{{state}} میں دیگر قابل ذکر کالجز',
        noCollegesForState: '{{state}} کے لیے کوئی کالج نہیں ملا۔',
        promptSelectState: 'براہ کرم شروع کرنے کے لیے ڈراپ ڈاؤن سے ریاست منتخب کریں۔'
      },
      visualizer: {
        title: 'ڈائنامک کیریئر پاتھ ویزؤلائزر',
        subtitle: 'راستہ نمایاں کرنے کے لیے نوڈ پر ہوور کریں۔ مزید جاننے کے لیے کلک کریں!',
        legendSubjects: 'مضامین',
        legendDegrees: 'ڈگریاں',
        legendCareers: 'کیریئرز'
      },
      skills: {
        title: 'اسکل بلڈر – اہم چیزیں سیکھیں',
        subtitle: 'سرفہرست کیریئرز کے لیے سب سے زیادہ طلب والی مہارتیں۔ مزید جاننے کے لیے کلک کریں۔',
        items: {
          problemSolving: { name: 'مسئلہ حل کرنا', description: 'انجینئرنگ اور ٹیک کرداروں کے لیے ضروری۔' },
          communication: { name: 'مواصلات', description: 'ٹیم ورک اور قیادت کے لیے اہم۔' },
          dsa: { name: 'ڈیٹا اسٹرکچرز اور الگورتھم', description: 'کمپیوٹر سائنس کی بنیاد۔' },
          projectManagement: { name: 'پراجیکٹ مینجمنٹ', description: 'پراجیکٹس کی منصوبہ بندی اور مؤثر عمل درآمد سیکھیں۔' },
          cloud: { name: 'کلاؤڈ کمپیوٹنگ (AWS/Azure)', description: 'جدید ٹیک منظرنامے میں زیادہ طلب۔' }
        }
      },
      timeline: {
        title: 'داخلہ اور اسکالرشپ ٹائم لائن',
        subtitle: 'امتحانات، داخلوں اور اسکالرشپس کی اہم آخری تاریخوں سے باخبر رہیں۔',
        filters: {
          all: 'سب',
          exams: 'امتحانات',
          scholarships: 'اسکالرشپس',
          admissions: 'داخلے'
        },
        searchPlaceholder: 'ایونٹ تلاش کریں...',
        errorFetch: 'ایونٹس حاصل نہیں کیے جا سکے۔ براہ کرم بعد میں کوشش کریں۔',
        status: {
          closed: 'بند',
          active: 'فعال',
          upcoming: 'آنے والا'
        },
        deadlineLabel: 'آخری تاریخ:',
        noEvents: 'آپ کی موجودہ تلاش یا فلٹر کے لیے کوئی ایونٹ نہیں ملا۔'
      }
    }
  }
};

const LANGUAGE_KEY = 'appLanguage';

export function setHtmlDirectionByLang(lang) {
  const isRtl = lang === 'ur';
  if (typeof document !== 'undefined') {
    document.documentElement.setAttribute('lang', lang);
    document.documentElement.setAttribute('dir', isRtl ? 'rtl' : 'ltr');
  }
}

const saved = typeof window !== 'undefined' ? localStorage.getItem(LANGUAGE_KEY) : null;
const defaultLang = saved || 'en';
setHtmlDirectionByLang(defaultLang);

i18n
  .use(initReactI18next)
  .init({
    resources,
    lng: defaultLang,
    fallbackLng: 'en',
    interpolation: { escapeValue: false },
    defaultNS: 'translation'
  });

export function changeLanguage(lang) {
  localStorage.setItem(LANGUAGE_KEY, lang);
  setHtmlDirectionByLang(lang);
  return i18n.changeLanguage(lang);
}

export default i18n;
