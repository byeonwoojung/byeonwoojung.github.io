module.exports = {
  title: `byeonwoojung-blog`,
  description: `byeonwoojung-blog`,
  language: `ko`, // `ko`, `en` => currently support versions for Korean and English
  siteUrl: `https://www.zoomkoding.com`,  // `https://www.zoomkoding.com` 건들지 말기
  ogImage: `/og-image.jpg`, // Path to your in the 'static' folder
  comments: {
    utterances: {
      repo: `https://github.com/byeonwoojung/byeonwoojung.github.io`, // `zoomkoding/zoomkoding-gatsby-blog`,
    },
  },
  ga: '0', // Google Analytics Tracking ID
  author: {
    name: `변우중`,
    bio: {
      role: `ML / AI 엔지니어`,
      description: ['오늘도 달리는', '내일이 달라지는'],
      thumbnail: 'profile.jpg', // Path to the image in the 'asset' folder
    },
    social: {
      github: `https://github.com/byeonwoojung`, // `https://github.com/zoomKoding`,
      linkedIn: ``, // `https://www.linkedin.com/in/jinhyeok-jeong-800871192`,
      email: `ricenuu.ds@gmail.com`, // `zoomkoding@gmail.com`,
    },
  },

  // metadata for About Page
  about: {
    timestamps: [
      // =====       [Timestamp Sample and Structure]      =====
      // ===== 🚫 Don't erase this sample (여기 지우지 마세요!) =====
      {
        date: '',
        activity: '',
        links: {
          github: '',
          post: '',
          googlePlay: '',
          appStore: '',
          demo: '',
        },
      },
      // ========================================================
      // ========================================================
      {
        date: '2025.10. ~',
        activity: '먹방 유튜버 방문 음식점 지도 서비스',
        links: {
          // post: 'https://byeonwoojung.github.io/about',
          // github: 'https://github.com/byeonwoojung',
          // demo: 'https://www.zoomkoding.com',
        },
      },
      {
        date: '2025.09. ~ 2025.11.',
        activity: `<독자 AI 파운데이션 모델 프로젝트>\n
        Agentic Tool Use 데이터 가공 구축 - 주식회사 플리토(프리랜서)`,
        links: {
          // post: 'https://byeonwoojung.github.io/about',
          // github: 'https://github.com/byeonwoojung',
          // demo: 'https://www.zoomkoding.com',
        },
      },
      {
        date: '2025.07. ~ 2025.09.',
        activity: '프랜차이즈 예비 창업자를 위한 AI 요약보고서 생성',
        links: {
          post: 'https://www.youtube.com/watch?v=aKPvZjedt4o',
          github: 'https://github.com/da-analysis/asac_8_dataanalysis',
          // demo: 'https://www.zoomkoding.com',
        },
      },
      {
        date: '2025.04. ~ 2025.05.\n2025.06. ~ 2025.07.',
        activity: '섬네일·제목·오디오 기반 통합적 유튜브 플레이리스트 조회수 예측',
        links: {
          // post: 'https://byeonwoojung.github.io/about',
          github: 'https://github.com/byeonwoojung/youtube-playlist-MLproject',
          // demo: 'https://www.zoomkoding.com',
        },
      },
    ],

    projects: [
      // =====        [Project Sample and Structure]        =====
      // ===== 🚫 Don't erase this sample (여기 지우지 마세요!)  =====
      {
        title: '',
        description: '',
        techStack: ['', ''],
        thumbnailUrl: '',
        links: {
          post: '',
          github: '',
          googlePlay: '',
          appStore: '',
          demo: '',
        },
      },
      // ========================================================
      // ========================================================
      {
        title: '[개발 중] 먹방 유튜버 방문 음식점 지도 서비스',
        description:
          '먹방 유튜버의 콘텐츠를 AI 기반 분석 및 분석 데이터 평가를 자동화하여, 방문 음식점명·위치·유튜버의 음식 리뷰 등의 정보를 지도에 나타냄으로써 사용자들에게 먹방 유튜버가 방문한 음식점 정보를 제공합니다. 사용자가 유튜버에게 직접 음식점 제안하는 기능을 통해 해당 유튜버에게 콘텐츠 아이디어에 관한 분석 데이터 제공을 목표로 합니다.',
        techStack: ['Python', 'Typescript'],
        thumbnailUrl: 'tzudong-home.png',
        links: {
          post: 'https://byeonwoojung.github.io/about',
          // github: 'https://github.com/byeonwoojung',
          // demo: 'https://www.zoomkoding.com',
        },
      },
      {
        title: '프랜차이즈 예비 창업자를 위한 AI 요약보고서 생성',
        description:
          '기존 서비스에서 프랜차이즈 예비창업자들의 브랜드 니즈 맞춤형 추천이 부족하고, 브랜드의 강점과 리스크 등을 한눈에 보기 어려웠습니다. 때문에 정보공개서(매출 등), 한국부동산원(지역, 층수, 상가규모별 임대료), 행정안전부(개·폐업정보) 등 5곳의 데이터를 통합하여 사용자 맞춤형 브랜드 추천 요약보고서를 제공하고자 합니다.',
        techStack: ['Python', 'RAG', 'PySpark', 'SQL', 'Langchain', 'OpenAI API', 'Naver Search API'],
        thumbnailUrl: 'franchise-main.png',
        links: {
          post: 'https://www.youtube.com/watch?v=aKPvZjedt4o',
          github: 'https://github.com/da-analysis/asac_8_dataanalysis',
          // demo: 'https://www.zoomkoding.com',
        },
      },
      // {
      //   title: '',
      //   description:
      //     '',
      //   techStack: [''],
      //   thumbnailUrl: 'tzudong-home.png',
      //   links: {
      //     // post: 'https://byeonwoojung.github.io/about',
      //     // github: 'https://github.com/byeonwoojung',
      //     // demo: 'https://www.zoomkoding.com',
      //   },
      // },
      {
        title: '섬네일·제목·오디오 기반 통합적 유튜브 플레이리스트 조회수 예측',
        description:
          '평소 즐겨 소비하는 플레이리스트 콘텐츠에 대해 테마별로 조회수가 높은 콘셉트가 있는 것으로 보여, 관심 있던 유튜브 감성·일상적인 플레이리스트 콘텐츠에 대해 조회수에 주요한 요소를 분석하고자 합니다. 플레이리스트 콘텐츠가 유튜브에서 트렌드를 주도하고 있으나 관련 연구 부족했습니다. 이에, 본 프로젝트는 ML 모델링과 데이터 분석을 통해 플레이리스트 콘텐츠 제작자의 섬네일·제목·오디오 선택에 지원하는 것을 목표로 합니다.',
        techStack: ['Python', 'Scikit-learn', 'PyTorch', 'TensorFlow', 'YuNet', 'Google Vision API', 'OpenCV', 'KMeans', 'OpenAI API', 'TfidfVectorizer', 'Wav2Vec2', 'Librosa', 'Pandas', 'Numpy'],
        thumbnailUrl: 'ml-paper.png',
        links: {
          // post: 'https://byeonwoojung.github.io/about',
          github: 'https://github.com/byeonwoojung/youtube-playlist-MLproject',
          // demo: 'https://www.zoomkoding.com',
        },
      },
    ],
  },
};
