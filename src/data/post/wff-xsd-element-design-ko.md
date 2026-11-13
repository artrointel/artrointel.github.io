---
title: 'WFF XML 스키마 설계 회고 1 — Scene graph와 요소·타입 설계'
excerpt: 'Scene graph의 XML 표현과 추상 Part의 타입 확장을 살펴봅니다. 초기 레이아웃 제약과 Part 명명에 관한 설계 회고를 함께 정리합니다.'
author: 'Woohyun Kim'
category: 'Platform Engineering'
tags:
  - 'Wear OS'
  - 'WFF'
  - 'XML'
  - 'XSD'
  - 'Schema Design'
language: ko
draft: true
metadata:
  robots:
    index: false
    follow: false
---

> 리뷰 중인 초안입니다. 공통 타입의 책임과 구체적인 요소 설계 이유는 후속 리뷰에서 보강합니다.

이 글은 [시리즈 머리말](/ko/wff-xsd-design-rationale-ko)에서 소개한 선언형 구조를 구체적으로 살펴봅니다. 렌더링 모델을 XML의 요소 계층과 XSD의 타입 구조에 어떻게 대응시켰는지, 초기 설계에 어떤 제약이 있었는지를 정리합니다.

## 설계 목표: 렌더링 장면과 동작의 선언형 명세

목표는 워치페이스의 렌더링 장면(scene)과 기능·동작을 XML로 명세하는 것이었습니다. 여기서 scene은 최종 출력 이미지가 아니라, 렌더링할 요소와 그 구성 관계를 기술하는 모델을 뜻합니다.

어떤 렌더링 노드로 scene을 구성할지, 각 요소가 어떤 기능과 동작을 제공할지를 선언할 수 있어야 했습니다. 따라서 설계 대상에는 시각 요소의 계층 구조와 함께 워치페이스의 동작을 표현하는 선언도 포함됐습니다.

현재 [Android의 WFF 소개](https://developer.android.com/training/wearables/wff)도 WFF를 워치페이스의 외형과 동작을 기술하는 선언형 XML 포맷으로 설명합니다.

### 초기 제약: 화면비에 따른 적응형 레이아웃

하드웨어마다 물리 디스플레이의 해상도와 화면비(aspect ratio)는 달라질 수 있습니다. 그러나 초기 설계에서는 화면비에 따라 노드 간 배치 관계를 재구성하는 적응형 레이아웃을 명세하지 않았습니다.

예를 들어 1:1 디스플레이와 다른 화면비의 디스플레이에서 노드를 서로 다른 관계로 재배치하도록 선언하는 기능은 초기 범위에 포함하지 않았습니다. 좌표와 크기로 노드의 기하 정보를 기술하는 것과, 디스플레이 조건에 따라 배치 관계를 결정하는 레이아웃 규칙은 별개의 설계 대상입니다.

공개 버전 1의 [`watchface.xsd`](https://github.com/google/watchface/blob/bbc2def58c94b18c4deec12d079b2792c2a33fef/third_party/wff/specification/documents/1/watchface.xsd)는 `width`와 `height`로 가상 캔버스의 크기를 정의합니다. 하위 요소의 위치와 크기는 이 가상 좌표계를 기준으로 기술됩니다. 물리 디스플레이 해상도, 가상 캔버스의 크기, 노드의 레이아웃 규칙은 구분해야 합니다. 초기 설계의 한계는 **좌표 표현의 부재가 아니라, 화면비 변화에 따른 배치 관계를 선언할 수 없었다는 점**입니다.

당시에는 초기 단계부터 여러 제조사의 다양한 화면비와 그에 따른 레이아웃 재배치를 제작자가 모두 고려하도록 하는 것이 합리적이지 않다고 판단했습니다. 우선 초기 포맷의 지원 범위를 한정하고, 필요해지면 후속 버전에서 레이아웃 모델을 확장할 수 있다고 보았습니다.

이는 초기 복잡도를 제한하기 위한 선택이면서 표현력의 한계이기도 했습니다. 후속 확장은 당시의 가능성 판단이었으며, 이 글에서 실제 구현되었다고 주장하는 것은 아닙니다.

## 구조 모델: Scene graph와 XML 요소 계층

기존 XML 기반을 이어간 이유 중 하나는 XML의 요소 트리가 워치페이스의 scene graph를 표현하기에 적합했기 때문입니다. 이 설계에서 scene graph는 렌더링 노드를 부모·자식 관계로 구성하는 계층 모델입니다.

전통적인 scene graph의 계층 모델을 XML 요소의 중첩 구조에 대응시켰습니다. `Scene` 아래에 렌더링 노드를 선언하고 `Group`으로 하위 노드를 묶으면, 장면의 구성 관계를 문서 구조로 표현할 수 있었습니다. 여기서 XML의 트리 표현을 선택했다는 것과 런타임이 DOM(Document Object Model) 파서를 사용한다는 것은 별개의 문제입니다. 이 글은 포맷의 구조 모델을 다룹니다.

아래는 공개 버전 1의 요소 이름을 사용해 관계를 단순화한 개념도입니다. 전체 문법이나 실행 가능한 워치페이스 예제는 아닙니다.

```text
WatchFace                 XML 문서의 루트 요소
└── Scene                 렌더링 장면의 컨테이너
    ├── Group             하위 렌더링 노드의 그룹
    │   ├── PartImage
    │   └── PartText
    └── PartDraw
```

공개 정의에서도 XML 문서의 루트 요소인 [`WatchFace`](https://github.com/google/watchface/blob/bbc2def58c94b18c4deec12d079b2792c2a33fef/third_party/wff/specification/documents/1/watchface.xsd)와 시각 요소의 컨테이너인 [`Scene`](https://github.com/google/watchface/blob/bbc2def58c94b18c4deec12d079b2792c2a33fef/third_party/wff/specification/documents/1/sceneElement.xsd)을 구분합니다. 렌더링 장면의 계층과 그 장면을 포함하는 XML 문서의 구조를 구분한 것입니다.

XML이 노드의 구성 관계를 표현한다면, XSD는 허용되는 요소·속성·타입과 구성 제약을 정의합니다. 장면의 계층 표현과 그 표현을 제한하는 스키마를 함께 설계할 수 있다는 점이 이 작업에 적합했습니다.

## 타입 설계: 추상 Part와 구체 타입의 확장

Part의 공통 정의를 추상 타입으로 분리하고, 이를 상속·확장하는 구체 타입을 정의하도록 설계했습니다. 대표적인 구체 요소가 `PartDraw`, `PartImage`, `PartText`입니다. XML 문서에서 노드를 중첩하는 포함 관계와 XSD에서 타입을 파생하는 상속 관계는 구분됩니다.

공개 XSD에서 [`AbstractPartType`](https://github.com/google/watchface/blob/bbc2def58c94b18c4deec12d079b2792c2a33fef/third_party/wff/specification/documents/1/group/part/abstractPartType.xsd)은 `xs:complexType`의 `abstract="true"`로 선언됩니다. [`PartDraw`](https://github.com/google/watchface/blob/bbc2def58c94b18c4deec12d079b2792c2a33fef/third_party/wff/specification/documents/1/group/part/draw/partDrawElement.xsd) 요소의 타입은 `xs:complexContent` 아래에서 `xs:extension base="AbstractPartType"`을 사용합니다. 공통 타입을 확장해 구체 Part의 구조를 정의하는 관계를 확인할 수 있습니다.

> **후속 보강:** 공통 정의에 어떤 책임을 넣었고 구체 Part에 무엇을 맡겼는지, 실제 설계 이유를 예시와 함께 추가할 예정입니다.

## 회고: `PartImage`인가, `ImagePart`인가

여러 설계 판단 중 지금도 기억에 남는 작은 사례가 Part의 명명 규칙입니다.

요소의 명명 규칙으로는 구체 Part에 공통 접두어 `Part`를 사용하도록 제안했습니다. 워치페이스 XML을 직접 작성하는 개발자가 관련 요소를 탐색하고 발견하는 과정, 즉 포맷의 discoverability를 고려한 결정이었습니다.

개발자가 ‘사용할 수 있는 Part에는 무엇이 있을까?’라고 생각하며 `Part`를 입력하면, IDE가 관련 요소를 함께 제안하는 흐름을 기대했습니다.

```text
Part
├── PartDraw
├── PartImage
└── PartText
```

이 목록은 명명 의도를 설명하는 예시입니다. 스키마 기반 코드 완성을 지원하는 IDE에서 공통 접두어가 후보 탐색의 단서가 되도록 하려는 판단이었으며, 특정 IDE의 자동완성 결과를 재현한 것은 아닙니다.

저를 포함한 네 명의 개발 담당자가 이 제안과 이유를 합리적이라고 판단해 해당 명명 방식을 채택했습니다.

동시에 `DrawPart`, `ImagePart`, `TextPart`가 영어권 사용자에게는 더 자연스럽게 읽힐 수도 있겠다는 의문이 남았습니다. 실제 사용자 조사를 통해 확인한 결과는 아니었지만, 개발자의 입력 흐름과 이름의 언어적 자연스러움 사이에서 고민했던 부분입니다.

지금 돌아보면 이 결정은 요소의 의미뿐 아니라 개발자가 포맷을 탐색하고 작성하는 과정까지 고려했던 사례입니다. 어느 명명 방식이 언제나 더 낫다고 결론 내리기보다, 당시 무엇을 우선했는지 남겨두고 싶습니다.

## 다음 보강 범위

추상 Part에 공통 정의를 배치한 기준과 구체 Part로 분리한 책임을 실제 요소를 따라 보강할 예정입니다. Part 접두어는 회고 중 하나이며, 다른 사례가 정리되면 추가합니다. XSD 작성 규칙 전반은 [coding convention 편](/ko/wff-xsd-coding-convention-ko)에서 다룹니다.

## 시리즈 목차

- [머리말: 내부 위키 명세에서 XSD로](/ko/wff-xsd-design-rationale-ko)
- [1. 요소·타입 설계: Scene graph와 Part](/ko/wff-xsd-element-design-ko)
- [2. Coding convention: XSD 정의의 일관성](/ko/wff-xsd-coding-convention-ko)
- [3. Validation: 목적과 검증 경계](/ko/wff-xsd-validation-ko)
