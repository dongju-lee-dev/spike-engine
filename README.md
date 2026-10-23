# spike-engine

SNN에 대해서 배우기 위해서 이 엔진을 개발한다.  
SNN 모델을 학습과 추론을 진행할 수 있어야 한다.  
SNN 모델을 만드는 것이 간단해야 한다.  

  ──────
  ### 모듈 1: Core (인프라 & 실행 제어)

  모든 모듈의 기반이 되는 메모리, 벡터화 보조, 데이터 변환, 엔진 루너를 담당합니다.  

  [o] 64바이트 Aligned Memory (memory.hpp):  
      • AVX-512 및 캐시 라인 기준 64바이트 메모리 정렬 할당자(AlignedAllocator<T, 64>) 구현.  
      • 기본 버퍼 타입 정의 (using FloatBuffer = std::vector<float, AlignedAllocator<float, 64>>;).  
  [o] 컴파일러 힌트 매크로 (compiler.hpp):  
      • Clang/GCC 호환 RESTRICT (__restrict__), 메모리 정렬 힌트(ASSUME_ALIGNED) 매크로.  
  [ ] 인코더/디코더 (encoder.hpp, decoder.hpp):  
      • Poisson Rate Encoder: 실수값([0.0, 1.0])을 확률적 스파이크(0.0f / 1.0f)로 실시간 변환.
      • Direct/Current Encoder: 실수값을 입력 전류 벡터로 직접 주입.  
      • Spike Rate Decoder: 출력 뉴런들의 누적 스파이크 횟수 집계 및 최대 발화 클래스 판별.  
  [ ] 엔진 오케스트레이터 (engine.hpp):  
      • Network와 LearningRule 목록을 소유.  
      • tick(input, reward, enable_learning = true): 추론/학습 제어.  
      • run(time_steps, ...): 다중 틱 연속 실행 제어.  

  ──────
  ### 모듈 2: Neuron (뉴런 군집, SoA)

  상태 변수들을 일렬 배열(SoA)로 관리하고, 자동 벡터화 루프로 1틱 전위 갱신 및 스파이크를 발생시킵니다.

  [o] 뉴런 베이스 인터페이스 (neuron.hpp):  
      • step(const FloatBuffer& current_in, FloatBuffer& spike_out) 순수 가상 함수.
      • reset(): 전위 및 내부 상태 초기화.  
  [o] IF 뉴런 (if_neuron.hpp):  
      • Integrate-and-Fire (누수 없음, 임계치 도달 시 발화 후 리셋).  
  [o] LIF 뉴런 (lif_neuron.hpp):  
      • Leaky Integrate-and-Fire (지수 감쇠 누수율 τ, 전위 적분, 리셋 전위).  
  [ ] ALIF 뉴런 (alif_neuron.hpp):  
      • Adaptive LIF (적응형 임계치 또는 억제 전류 w 적분, 피로도 반영).  
  [ ] Izhikevich 뉴런 (izhikevich_neuron.hpp):  
      • 생체 뉴런의 복잡한 동역학 재현 (v' = 0.04v² + 5v + 140 - u + I, u' = a(bv - u)).
      • 4칙 연산 위주로 구성되어 Clang 자동 벡터화에 매우 적합.

  ──────
  ### 모듈 3: Synapse (시냅스 연결 & 신호 전달)

  Pre-뉴런의 스파이크를 Post-뉴런의 입력 전류로 변환하여 전달합니다. 학습 기능은 갖지 않고 순수 가중치 및 전달만 처리합니다.

  [o] 시냅스 베이스 인터페이스 (synapse.hpp):  
      • forward(const FloatBuffer& pre_spikes, FloatBuffer& post_current)  
      • 가중치 읽기/쓰기 인터페이스 (get_weights(), set_weights()).  
  [o] 완전 연결 시냅스 (dense_synapse.hpp):  
      • 1차원 평탄화된 가중치 행렬 (M × N).  
      • Pre 뉴런이 발화한 경우 해당 가중치 행을 Post 전류 버퍼에 연속 메모리 벡터 덧셈.  
  [o] 희소 연결 시냅스 (sparse_synapse.hpp):  
      • 비어있는 0 연산 제거를 위한 평탄화된 인덱스/가중치 리스트(Flat CSR/Adjacency Array).
      • 발화된 뉴런에 연결된 유효 Post 뉴런들에만 선택적 전류 누적.  
  [o] 가중치 초기화 유틸리티 (weight_init.hpp):  
      • Uniform, Normal, Xavier/He 등 기초 분포 초기화기.  

  ──────
  ### 모듈 4: Network (토폴로지 & 전방향 추론)

  뉴런 군집들과 시냅스들을 등록하고, 연결 관계에 따라 1틱 동안 전방향(Forward) 추론을 수행합니다.

  [ ] 네트워크 컨테이너 (network.hpp):  
      • add_population<NeuronType>(size, config): 뉴런 군집 생성 및 추가.  
      • connect(pre_idx, post_idx, synapse_type, init_cfg): 뉴런 군집 간 시냅스 생성 및 연결.  
      • step(const FloatBuffer& input): 전체 뉴런/시냅스를 순회하며 추론 1틱 실행 (중간 버퍼 핑퐁 재사용).  
      • reset(): 모든 뉴런 및 내부 버퍼 상태 리셋.  

  ──────
  ### 모듈 5: Learning (사후 일괄 가소성 규칙)

  Network 외부에서 특정 Pre/Post 뉴런과 시냅스를 포인터로 참조하여 가중치를 갱신합니다.

  [ ] 학습 규칙 베이스 인터페이스 (learning.hpp):  
      • update(float reward = 0.0f)  
      • reset()  
  [ ] STDP (stdp.hpp):  
      • Pre-Trace (x) & Post-Trace (y) 지수 감쇠 갱신.  
      • LTP (장기 강화: Post 발화 시 Pre-Trace만큼 ΔW > 0).  
      • LTD (장기 억제: Pre 발화 시 Post-Trace만큼 ΔW < 0).  
      • 가중치 바운딩 ([wₘᵢₙ, wₘₐₓ]).  
  [ ] R-STDP (r_stdp.hpp):  
      • 적격성 흔적(Eligibility Trace, E) 배열 유지 (E[t] = E[t - 1] · exp(-dt/τ_c) + ΔW_{STDP}).  
      • 외부 보상(R) 신호 인가 시 최종 가중치 갱신 (ΔW = R · E).  

  ──────
  ## 3. 단계별 구현 로드맵

  Phase 1: 기반 인프라 (Core)  
      • memory.hpp, compiler.hpp 구현 및 메모리 정렬 검증.  
  Phase 2: 뉴런 구현 (Neuron)  
      • neuron_population.hpp 인터페이스 작성.  
      • lif_neuron.hpp (LIF SoA) 구현 및 Clang 벡터화 검증.  
  Phase 3: 시냅스 구현 (Synapse)  
      • dense_synapse.hpp 구현 (스파이크 기반 가중치 덧셈 벡터화).  
  Phase 4: 네트워크 & 엔진 파이프라인 (Network & Engine)  
      • network.hpp 전방향 전달 파이프라인 완성.  
      • engine.hpp 1틱 제어 연결.  
  Phase 5: 학습 규칙 (Learning)  
      • stdp.hpp 구현 및 시냅스 가중치 변화 확인.  
      • r_stdp.hpp 구현.  
  Phase 6: 확장 및 예제 (Expansion & Examples)  
      • 희소 시냅스(sparse_synapse.hpp), 추가 뉴런(IF, ALIF, Izhikevich) 구현.  
      • 실시간 인코더/디코더 추가 및 간단한 학습 예제 작성.  