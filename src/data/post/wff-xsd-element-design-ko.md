---
title: 'WFF XML 스키마 설계 회고 1 — 요소·속성·타입의 정의와 재사용'
excerpt: '워치페이스 XML을 XSD로 어떻게 정의할 것인가. Scene graph에서 출발해 렌더링·비렌더링 요소, 공통 속성과 도메인 타입을 분리하고 재사용한 설계를 살펴봅니다.'
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

워치페이스 XML을 XSD로 정의하려면 무엇부터 설계해야 할까요? [머리말](/ko/wff-xsd-design-rationale-ko)에서 소개한 작업은, 렌더링 scene과 동작을 XML의 어떤 요소·속성으로 표현하고 그 값과 구성 관계를 어떻게 정의할 것인가에서 시작했습니다.

이 글에서는 렌더링 계층을 XML로 표현하는 모델부터 살펴보고, **요소(element)·속성(attribute)·타입(type)의 정의를 분리하고 가능한 한 원자적인 단위로 재사용한다**는 설계 원칙이 실제 XSD에 어떻게 드러나는지 정리합니다. 코드 설명은 머리말과 같은 공개 버전 1을 기준으로 합니다.

## Scene graph를 XML 요소 계층으로 표현하기

렌더링 scene을 부모·자식 관계의 계층으로 구성하는 것은 전통적인 scene graph의 접근입니다. XML 역시 요소의 중첩으로 계층을 선언하므로, 워치페이스의 렌더링 모델을 표현하는 데 자연스럽게 대응시킬 수 있었습니다. 시계의 시각 요소를 계층으로 구성하는 모델을 XML로 옮기는 일 자체는 어렵지 않았습니다.

문서의 루트인 `WatchFace` 아래에 `Scene`을 두고, 그 하위에 렌더링 요소와 그룹을 선언합니다. 다음은 구성 관계를 단순화한 개념도입니다.

```text
WatchFace
└── Scene
    ├── PartImage
    ├── Group
    │   ├── PartDraw
    │   └── PartText
    └── AnalogClock
        ├── HourHand
        ├── MinuteHand
        └── SecondHand
```

이 모델에서는 **XML 인스턴스에 선언한 렌더링 요소의 순서가 scene을 순회하며 그리는 순서로 이어집니다.** 작성자는 XML의 계층과 선언 순서를 보면서 렌더링 구성을 이해할 수 있습니다. [Android의 렌더링 설명](https://developer.android.com/training/wearables/wff/effects)도 렌더러가 scene tree를 순서대로 순회하며 요소를 그리는 모델을 설명합니다.

여기서 말하는 순서는 워치페이스 XML 문서의 노드 순서입니다. XSD 파일 안에서 요소 정의를 나열한 순서가 렌더링 순서를 결정하는 것은 아닙니다. XSD는 어떤 요소와 속성을 사용할 수 있고 어떻게 구성할 수 있는지를 정의하며, 그 문서를 해석하고 렌더링하는 것은 Runtime의 역할입니다.

## 요소·속성·타입을 분리한 이유

워치페이스의 요소마다 필요한 정의를 그 자리에서 모두 작성할 수도 있습니다. 그러나 위치·크기·색상 같은 개념은 여러 요소에서 반복되고, 같은 데이터의 의미와 제약도 함께 유지되어야 합니다.

따라서 다음 세 층을 구분했습니다.

| 정의 단위 | 설계에서의 역할                                | 예시                                                       |
| --------- | ---------------------------------------------- | ---------------------------------------------------------- |
| 요소      | scene의 구성과 기능·동작을 선언                | `PartImage`, `HourHand`, `Transform`                       |
| 속성      | 요소의 위치·크기·렌더링 파라미터 등을 기술     | `x`, `y`, `width`, `height`, `alpha`                       |
| 타입      | 값의 표현과 허용 범위, 요소의 구성 규칙을 정의 | `dimensionType`, `argbHexadecimalType`, `AbstractPartType` |

핵심은 **공통 개념을 작게 정의하고 필요한 곳에서 참조·조합하는 것**이었습니다. 요소의 포함 관계, 속성의 사용, 값의 제약을 분리하면 같은 정의를 반복해서 작성하는 일을 줄이고 공통 규칙을 유지하기 쉬워집니다.

### XSD의 추상화와 타입 파생

XSD는 추상 요소와 추상 타입, 타입의 확장(extension)과 제약(restriction)을 지원합니다. 객체지향 설계에서 공통 정의를 추상화하고 상속하는 사고방식을 스키마 설계에도 적용할 수 있었습니다. 속성 선언과 속성 그룹, 데이터 타입 역시 이름을 부여해 재사용할 수 있습니다. 관련 개념은 [W3C XML Schema Primer](https://www.w3.org/TR/xmlschema-0/)에서도 설명합니다.

예를 들어 [AbstractPartType](https://github.com/google/watchface/blob/bbc2def58c94b18c4deec12d079b2792c2a33fef/third_party/wff/specification/documents/1/group/part/abstractPartType.xsd)은 `abstract="true"`인 복합 타입입니다. [PartDraw](https://github.com/google/watchface/blob/bbc2def58c94b18c4deec12d079b2792c2a33fef/third_party/wff/specification/documents/1/group/part/draw/partDrawElement.xsd) 요소의 타입은 `xs:complexContent` 안에서 `xs:extension base="AbstractPartType"`을 사용해 이를 확장합니다.

이 타입 파생 관계는 XML 문서에서 `Group` 아래에 `PartDraw`를 배치하는 포함 관계와는 별개입니다. 노드를 어떻게 구성할지와 그 노드의 공통 정의를 어떻게 재사용할지를 각각 설계한 것입니다.

## 요소: 렌더링 구성과 동작 정의

워치페이스의 요소는 역할에 따라 렌더링 계층을 구성하는 요소와, 그 요소의 동작을 기술하는 비렌더링 요소로 나누어 생각할 수 있었습니다. 이는 설계 의도를 설명하기 위한 분류입니다.

### 렌더링 계층을 구성하는 요소

디자인의 유연성을 위해 이미지·텍스트·도형을 표현하는 범용 요소를 두었습니다. 한편 워치페이스라는 도메인에서 반복적으로 필요한 시각 요소는 의미가 드러나는 요소로 정의했습니다.

| 구분                    | 대표 요소                              | 목적                                                          |
| ----------------------- | -------------------------------------- | ------------------------------------------------------------- |
| 범용 시각 요소          | `PartImage`, `PartText`, `PartDraw`    | 이미지·텍스트·도형을 조합해 다양한 디자인을 구성              |
| 계층 컨테이너           | `Group`                                | 하위 렌더링 노드를 묶고 계층 단위로 구성                      |
| 시계의 의미를 가진 요소 | `HourHand`, `MinuteHand`, `SecondHand` | 아날로그 시계의 시·분·초 바늘처럼 역할이 분명한 요소를 선언   |
| 시간 외 데이터의 표현   | `ComplicationSlot`, `Complication`     | 워치페이스에 제공되는 다른 데이터를 이미지·텍스트 등으로 표현 |

`Group`은 렌더링 계층의 일부이지만 직접 픽셀을 그리지는 않습니다. 공개 [groupElement.xsd](https://github.com/google/watchface/blob/bbc2def58c94b18c4deec12d079b2792c2a33fef/third_party/wff/specification/documents/1/group/groupElement.xsd)에서는 하위 계층의 위치·크기·각도·색상에 영향을 주는 컨테이너로 설명합니다.

범용 요소와 시계에 특화된 요소를 함께 두면 표현의 자유도와 도메인의 의미를 함께 담을 수 있습니다. 예를 들어 [HourHand](https://github.com/google/watchface/blob/bbc2def58c94b18c4deec12d079b2792c2a33fef/third_party/wff/specification/documents/1/clock/hourHand.xsd)는 시침을 표현하는 요소이며, [ComplicationSlot](https://github.com/google/watchface/blob/bbc2def58c94b18c4deec12d079b2792c2a33fef/third_party/wff/specification/documents/1/complication/complicationSlotElement.xsd)은 데이터 표시를 위한 슬롯과 하위 `Complication`을 구성합니다.

### 비렌더링 요소: 런타임 동작을 선언하기

`Transform`, `Gyro` 같은 요소는 자신이 렌더링할 이미지를 정의하는 대신, 부모 렌더링 요소가 런타임 데이터에 따라 어떻게 동작할지 기술합니다.

렌더링 요소에 직접 선언한 위치·크기·각도 등의 속성은 기본적인 상태를 기술합니다. 반면 하위의 동작 요소는 런타임의 동적 조건이나 데이터에 따라 그 상태를 어떻게 변경할지 선언합니다.

- [Transform](https://github.com/google/watchface/blob/bbc2def58c94b18c4deec12d079b2792c2a33fef/third_party/wff/specification/documents/1/common/transform/transformElements.xsd)은 `target`으로 대상 속성을 지정하고, `value`에 표현식을 기술합니다.
- [Gyro](https://github.com/google/watchface/blob/bbc2def58c94b18c4deec12d079b2792c2a33fef/third_party/wff/specification/documents/1/common/transform/gyroElements.xsd)는 센서 데이터를 사용하는 표현식으로 부모 `Group`이나 Part의 속성을 조정합니다.

따라서 XML의 자식 요소가 모두 별도의 시각 객체인 것은 아닙니다. 어떤 요소는 scene을 구성하고, 어떤 요소는 그 scene의 동작을 선언합니다. 이러한 구분을 통해 정적인 구성과 동적인 렌더링 동작을 같은 문서 안에서 기술할 수 있었습니다.

## 속성: 공통 선언과 attributeGroup의 재사용

속성은 요소의 배치와 렌더링 파라미터를 기술합니다. 여러 요소에서 공통으로 쓰는 속성은 선언과 값의 타입을 재사용하도록 구성했습니다.

예를 들어 [geometricAttributes.xsd](https://github.com/google/watchface/blob/bbc2def58c94b18c4deec12d079b2792c2a33fef/third_party/wff/specification/documents/1/common/attributes/geometricAttributes.xsd)는 `x`, `y`, `width`, `height`를 각각 `dimensionType`을 사용하는 속성으로 선언합니다. 이를 다시 위치와 크기의 그룹으로 묶고, 필요한 조합을 참조합니다.

```text
dimensionType
├── x, y 속성 선언
│   └── positionAttributes
└── width, height 속성 선언
    └── sizeAttributes

geometricAttributes
├── positionAttributes 참조
└── sizeAttributes 참조
```

이 파일에는 필수 속성으로 사용할 그룹도 별도로 정의되어 있습니다. `geometricAttributesRequired`는 위치와 크기의 필수 속성 그룹을 조합하며, `AbstractPartType`과 시계 바늘 같은 요소에서 참조합니다.

`attributeGroup`을 활용하면 여러 요소가 공유하는 속성 구성을 간결하게 표현할 수 있습니다. 개별 요소에서 같은 속성을 반복 선언하기보다, 공통 속성의 의미와 사용 조건을 한곳에서 살펴볼 수 있습니다.

속성의 타입도 요소마다 익명 타입으로 중복 정의하기보다 **이름 있는 공통 타입을 참조하는 방식**을 우선했습니다. 반복되는 값의 제약을 따로 유지하지 않으려는 원칙이었습니다. 공개 정의에는 개별 용도의 익명 타입도 있으므로, 모든 속성에 예외 없이 적용한 규칙으로 설명하지는 않습니다.

## 타입: WFF 도메인의 값과 제약을 정의하기

XSD는 `xs:string`, `xs:integer`, `xs:float` 같은 내장 타입을 제공합니다. 필요한 곳에서는 이를 그대로 사용하고, WFF의 도메인에 맞는 값이나 제약이 필요하면 별도의 타입으로 정의했습니다.

### sourceType: 도메인별 값을 정의하고 조합하기

[sourceType.xsd](https://github.com/google/watchface/blob/bbc2def58c94b18c4deec12d079b2792c2a33fef/third_party/wff/specification/documents/1/common/attributes/sourceType.xsd)는 시간·언어·배터리·센서·건강 데이터 등의 source를 타입으로 나누어 정의합니다.

예를 들어 `languageSourceType`은 `xs:string`을 기반으로 언어 관련 식별자를 `xs:enumeration`으로 제한합니다. 시간 관련 source는 `timeUnitSourceType`과 `_timeSourceType`으로 나누고 `timeSourceType`에서 합칩니다. 마지막으로 `sourceType`이 여러 도메인의 타입을 조합합니다.

공개 정의의 타입 조합 관계를 요약하면 다음과 같습니다.

```text
timeUnitSourceType + _timeSourceType
└── timeSourceType

timeSourceType, languageSourceType, batterySourceType, ...
└── sourceType  (xs:union)
```

값의 집합을 하나의 큰 목록으로만 유지하는 대신, 도메인별로 정의한 타입을 조합할 수 있는 구조입니다. 이는 원자적인 정의를 재사용한다는 원칙이 값의 타입에도 적용된 예입니다.

### argbHexadecimalType: 내장 타입에 도메인 제약 추가하기

[colorAttributes.xsd](https://github.com/google/watchface/blob/bbc2def58c94b18c4deec12d079b2792c2a33fef/third_party/wff/specification/documents/1/common/attributes/colorAttributes.xsd)의 `argbHexadecimalType`은 `xs:string`에서 파생한 타입입니다. `xs:restriction` 안에 `xs:pattern`을 두어 `#AARRGGBB` 또는 `#RRGGBB` 색상 표기를 위한 패턴 제약을 정의합니다.

문자열이라는 표현 위에 WFF가 다루는 색상 값의 형식을 명시한 것입니다. 이때 내장 타입 자체를 변경하는 것이 아니라, 그 타입을 기반으로 도메인에 맞는 제약을 추가한 타입을 만듭니다.

타입을 분리하면 속성은 그 값을 어디에 사용하는지 기술하고, 타입 정의는 어떤 값을 기대하는지 설명할 수 있습니다. 복합 타입의 확장, 속성 그룹의 조합, 단순 타입의 제약을 함께 사용하며 WFF의 정의를 구체화했습니다.

## 초기 설계의 범위와 회고

요소·속성·타입의 재사용 구조를 정하는 것과 함께, 초기 포맷이 어디까지 표현할 것인지도 결정해야 했습니다. 다음은 앞선 리뷰에서 정리한 레이아웃 제약과 Part 명명에 관한 회고입니다.

### 초기 제약: 화면비에 따른 적응형 레이아웃

하드웨어마다 물리 디스플레이의 해상도와 화면비(aspect ratio)는 달라질 수 있습니다. 그러나 초기 설계에서는 화면비에 따라 노드 간 배치 관계를 재구성하는 적응형 레이아웃을 명세하지 않았습니다.

예를 들어 1:1 디스플레이와 다른 화면비의 디스플레이에서 노드를 서로 다른 관계로 재배치하도록 선언하는 기능은 초기 범위에 포함하지 않았습니다. 좌표와 크기로 노드의 기하 정보를 기술하는 것과, 디스플레이 조건에 따라 배치 관계를 결정하는 레이아웃 규칙은 별개의 설계 대상입니다.

공개 버전 1의 [`watchface.xsd`](https://github.com/google/watchface/blob/bbc2def58c94b18c4deec12d079b2792c2a33fef/third_party/wff/specification/documents/1/watchface.xsd)는 `width`와 `height`로 가상 캔버스의 크기를 정의합니다. 하위 요소의 위치와 크기는 이 가상 좌표계를 기준으로 기술됩니다. 물리 디스플레이 해상도, 가상 캔버스의 크기, 노드의 레이아웃 규칙은 구분해야 합니다. 초기 설계의 한계는 **좌표 표현의 부재가 아니라, 화면비 변화에 따른 배치 관계를 선언할 수 없었다는 점**입니다.

당시에는 초기 단계부터 여러 제조사의 다양한 화면비와 그에 따른 레이아웃 재배치를 제작자가 모두 고려하도록 하는 것이 합리적이지 않다고 판단했습니다. 우선 초기 포맷의 지원 범위를 한정하고, 필요해지면 후속 버전에서 레이아웃 모델을 확장할 수 있다고 보았습니다.

이는 초기 복잡도를 제한하기 위한 선택이면서 표현력의 한계이기도 했습니다. 후속 확장은 당시의 가능성 판단이었으며, 이 글에서 실제 구현되었다고 주장하는 것은 아닙니다.

### 명명 회고: `PartImage`인가, `ImagePart`인가

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

팀에서도 이 제안의 이유를 합리적이라고 보아 해당 명명 방식을 채택했습니다.

동시에 `DrawPart`, `ImagePart`, `TextPart`가 영어권 사용자에게는 더 자연스럽게 읽힐 수도 있겠다는 의문이 남았습니다. 실제 사용자 조사를 통해 확인한 결과는 아니었지만, 개발자의 입력 흐름과 이름의 언어적 자연스러움 사이에서 고민했던 부분입니다.

지금 돌아보면 이 결정은 요소의 의미뿐 아니라 개발자가 포맷을 탐색하고 작성하는 과정까지 고려했던 사례입니다. 어느 명명 방식이 언제나 더 낫다고 결론 내리기보다, 당시 무엇을 우선했는지 남겨두고 싶습니다.

## 다음 리뷰에서 구체화할 내용

이 글에서는 scene의 계층 표현에서 출발해 요소·속성·타입을 나누고 재사용하는 큰 틀을 정리했습니다. 다음 리뷰에서는 실제 요소 정의를 따라 추상 타입의 공통 책임, 속성 재사용의 선택 기준과 예외를 더 구체화하려 합니다.

정의를 일관되게 작성하기 위한 규칙은 [coding convention 편](/ko/wff-xsd-coding-convention-ko)에서, 스키마로 보장할 수 있는 범위와 한계는 [Validation 편](/ko/wff-xsd-validation-ko)에서 이어갑니다.

## 시리즈 목차

- [머리말](/ko/wff-xsd-design-rationale-ko)
- [1. 요소·속성·타입 설계](/ko/wff-xsd-element-design-ko)
- [2. Coding convention](/ko/wff-xsd-coding-convention-ko)
- [3. Validation](/ko/wff-xsd-validation-ko)
