---
layout: post                          # (require) default post layout
title: "Protocol Buffer"   # (require) a string title
date: 2023-01-20       # (require) a post date
categories: [ProcessCommunication]          # (custom) some categories, but make sure these categories already exists inside path of `category/`
tags: [ProcessCommunication]                      # (custom) tags only for meta `property="article:tag"`
---

# Protocol Buffers

Protocol buffers는 IDL로서 하나의 언어이다. 이 언어를 사용하여 protocol interface를 구현하려면 숙지해야하는 내용들을 정리했다.

(note: `.proto` 파일에서의 comment는 `// blahblah` 또는 `/* blahblah */` syntax로 작성할 수 있다.)

## fields

message에 포함하려는 data를 field로 define할 수 있다. 그리고 각 field는 하나의 name과 하나의 type, 그리고 unique number를 가지고 있다.

### field types

크게 두 가지로 나뉨:

- scalars(e.g., integers or string) 
- composite(including enumeration 외에도 other message type들이 포함된 composite형태) 

### unique number

binary format의 message에서 field를 식별하기위해 각 field에 unique number를 지정해준다. (so must not be changed once the message type is in use)

field number로 사용할 수 있는 숫자 range :  1부터 2^29 -1(=536,870,911)까지

field number range별 차이:

- 1~15 : take one byte to encode

  1~15사이의 field number는 frequently occurring message element에 assign해야 함.

- 16~2047 :  take two bytes to encode

  단, 19000 ~ 19999 사이의 숫자는 사용할 수 없다. (`FieldDescriptor::kFirstReservedNumber` through `FieldDescriptor::kLastReservedNumber`), as they are reserved for the Protocol Buffers implementation.)

<br>

"SearchRequest"라는 message format을 define한다고 가정하면, 각 search request가 다음 내용을 포함하게하도록 다음과 같이 구현한다. 3개의 fields를 가지고있다.

- query string
- particular page of result you are interested in
- number of results per page

```protobuf
// proto3를 사용한다고 명시하기.(syntax를 명시하지 않으면 default는 "proto2"다.)
syntax = "proto3";

// SearchRequest라는 이름의 message definition. 3개의 fields를 가지고 있음.
message SearchRequest {
  string query = 1;
  int32 page_number = 2;
  int32 result_per_page = 3;
}
```

### field rules

message fields가 될 수 있는 형태:

- `singular`

  message에서 singular field는 아얘 없거나 있다면 한개까지만 존재할 수 있음. 다른 field rule이 따로 명시되어 있지않으면, 이 rule이 default rule이다.

- `optional`

  `singular`와 같은 rule이 적용 되지만, value가 설정된다. 다음과 같은 두 가지 states가 있음:

  - the field is set: field가 explicitly set된 value 또는 wire에서 parse된 value를 가지고 있음. wire에 serialize된다.
  - the field is unset: field가 default value를 반환한다. wire에 serialize되지 않는다.

- `repeated`

  여러번 반복될 수 있다. (this field type can be repeated zero or more times in a well-formed message) 순서가 preserve된다.

  repeated fields of scalar numeric type들은 default로 `packed` encoding을 사용한다.

- `map`

  paired key/value field type이다. 

`reserved` keyword를 통해 delete된 field의 field number를 유지할 수 있다. 

<br>

<Br>

## proto 설정 방법

`.proto` 파일을 생성 위치(file location)은 root directory아래 `proto`라는 subpackage를 형성하여 `.proto`파일들만 따로 모아두는것이 더 적합하다. (e.g., `root_dir/proto/_myproto.proto`) 다른 lanauge sources를과 함께 같은 directory에 넣는것은 바람직하지 못하다.

하나의 `.proto` 파일안에 여러개의 message type들이 define될 수 있다. multiple message들이 연관되어 있는 경우 하나의 proto 파일안에 define되는것이 유용하다. 

`.proto` 파일에 protocol buffer compiler를 실행하면, compiler가 선택한 언어(e.g., Python)로 `.proto`에 define된 message type들이 활용될 수 있도록 code를 생성한다. 이 과정에서 field 값들이 만들어지고, message들이 output stream으로 serialize되고 input stream으로부터 message들이 parsing된다.

Python의 경우, the Python compiler generates a module with a static descriptor of each message type in your `.proto`, which is then used with a *metaclass* to create the necessary Python data access class at runtime.

### scalar value types

`.proto`에서 define된 .proto Type이 다음과 같이 선택한 언어의 corresponding data type (e.g., Python Type)으로 자동으로 생성된다. 다음 테이블 참고:

| .proto  Type | Python Type                     | Notes                                                        |
| ------------ | ------------------------------- | ------------------------------------------------------------ |
| double       | float                           |                                                              |
| float        | float                           |                                                              |
| int32        | int                             | Uses variable-length encoding.  Inefficient for encoding negative numbers – if your field is likely to have  negative values, use sint32 instead. |
| int64        | int/long[4]                     | Uses variable-length encoding.  Inefficient for encoding negative numbers – if your field is likely to have  negative values, use sint64 instead. |
| uint32       | int/long[4]                     | Uses variable-length encoding.                               |
| uint64       | int/long[4]                     | Uses variable-length encoding.                               |
| sint32       | int                             | Uses variable-length encoding. Signed  int value. These more efficiently encode negative numbers than regular  int32s. |
| sint64       | int/long[4]                     | Uses variable-length encoding. Signed  int value. These more efficiently encode negative numbers than regular  int64s. |
| fixed32      | int/long[4]                     | Always four bytes. More efficient  than uint32 if values are often greater than 228. |
| fixed64      | int/long[4]                     | Always eight bytes. More efficient  than uint64 if values are often greater than 256. |
| sfixed32     | int                             | Always four bytes.                                           |
| sfixed64     | int/long[4]                     | Always eight bytes.                                          |
| bool         | bool                            |                                                              |
| string       | str/unicode[5]                  | A string must always contain UTF-8  encoded or 7-bit ASCII text, and cannot be longer than 232. |
| bytes        | str (Python 2)/bytes (Python 3) | May contain any arbitrary sequence of  bytes no longer than 232. |

notes on Python types : 64-bit or unsigned 32-bit integers are always represented as long when decoded, but can be an int if an int is given when setting the field. In all cases, the value must fit in the type represented when set. 

### default values

message가 parse되었을때에, encoded message에 particular singular element가 없다면, 해당 field는 default value가 주어진다. 여기 default는 다음과 같이 설정된다.

- For strings, the default value is the empty string.
- For bytes, the default value is empty bytes.
- For bools, the default value is false.
- For numeric types, the default value is zero.
- For [enums](https://protobuf.dev/programming-guides/proto3/#enum), the default value is the **first defined enum value**, which must be 0.
- For message fields, the field is not set. Its exact value is language-dependent. See the [generated code guide](https://protobuf.dev/reference/) for details.

### enumeration

message type을 define할때, message의 field가 특정 카테고리 값들 중 하나로 지정되도록 제한되기를 원한다면, enumeration을 활용할 수 있다. 여기서 주의할 점은 모든 `enum` definition에서 첫 번째 constant는 default로 0을 mapping할 constant가 되어야 한다.

- use 0 as numeric default value
- 0 value must be the first element

e.g.,  `SearchRequest` message의 `corpus` 라는 field가  `A`, `B`, `C`, `D`,`E` 이렇게 5개중 하나가 되도록 define하려면 다음과 같이 define한다. Define default as its first element 'UNSPECIFIED'.

```protobuf
// 직접 설정하는 Corpus type의 field 정보
enum Corpus{
    CORPUS_UNSPECIFIED = 0;
    CORPUS_A = 1;
    CORPUS_B = 2;
    CORPUS_C = 3;
    CORPUS_D = 4;
    CORPUS_E = 5;
}

message SearchRequest {
    string query = 1;
    int32 page_number = 2;
    int32 result_per_page = 3;
    Corpus corpus = 4;
}
```

### enum alias의 사용

You can define aliases by assigning the same value to different enum constants. To do this you need to set the `allow_alias` option to `true`. 이렇게 하지않으면 warning message가 뜸.

enumerator constant는 32-bit integer의 range내의 값으로 설정되어야 한다. `enum` value들은 wire에서 varint encoding을 사용하기때문에, negative values들은 inefficient함. (thus not recommended) 

message definition내에서 `enum`들을 define할 수 있다. 하나의 message내에서 define된 `enum` type을 다른 message에서 field의 type으로 사용할수도 있다. (using the syntax `_MessageType_._EnumType_`.)

```protobuf
enum EnumAllowingAlias {
  option allow_alias = true;
  EAA_UNSPECIFIED = 0;
  EAA_STARTED = 1;
  EAA_RUNNING = 1;
  EAA_FINISHED = 2;
}

enum EnumNotAllowingAlias {
  ENAA_UNSPECIFIED = 0;
  ENAA_STARTED = 1;
  // ENAA_RUNNING = 1;  // Uncommenting this line will cause a warning message.
  ENAA_FINISHED = 2;
}
```

### other message types 사용

다른 message types를 field types로 사용할수도 있다. 

e.g., `Result` messages를 각 `SearchResponse` message에 포함시키고 싶다면, 같은 `.proto` 파일에서 `Result` message type을 define하고 `SearchResponse` 안에서 field type `Result`를 specify하면 된다. 

code 구현:

```protobuf
message SearchResponse {
  repeated Result results = 1;
}

message Result {
  string url = 1;
  string title = 2;
  repeated string snippets = 3;
}
```

### nested types

다른 message types안에서 message types를 define하고 사용할 수 있다. 

```protobuf
message SearchResponse {
  message Result {
    string url = 1;
    string title = 2;
    repeated string snippets = 3;
  }
  repeated Result results = 1;
}
```

이렇게 nested type으로 define된 message type을 message 밖에서 사용하고 싶다면, 다음과 같이 syntax `_Parent_._Type_`로 사용할 수 있다.

```protobuf
message SomeOtherMessage {
  SearchResponse.Result result = 1;
}
```

nesting에는 depth가 제한되어있지 않아서 다음과 같이 nest안에 nest된 message type의 definition도 가능하다.

```protobuf
message Outer {                  // Level 0
  message MiddleAA {  // Level 1
    message Inner {   // Level 2
      int64 ival = 1;
      bool  booly = 2;
    }
  }
  message MiddleBB {  // Level 1
    message Inner {   // Level 2
      int32 ival = 1;
      bool  booly = 2;
    }
  }
}
```

### unknown fields

parser이 인지하지 못하는 data이 경우, unknown field가 data를 represent한다. 

for example, when an old binary parses data sent by a new binary with new fields, those new fields become unknown fields in the old binary.

proto 3.5 이후 버젼에서부터는 unknown fields가 discard되지 않고 retain되어 serialized output에 포함된다.

### any

"any"는 message type중 하나로, `.proto`에서의 definition 없이, embedded type으로 message를 활용할 수 있도록 해준다. any를 사용하기 위해서는 `google/protobuf/any.proto`를 import 해야한다.

An `Any` contains an arbitrary serialized message as `bytes`, along with a URL that acts as a globally unique identifier for and resolves to that message’s type. The default type URL for a given message type is `type.googleapis.com/_packagename_._messagename_`.

```protobuf
import "google/protobuf/any.proto";

message ErrorStatus {
  string message = 1;
  repeated google.protobuf.Any details = 2;
}
```

### oneof

하나의 message안에 여러개의 fields가 있는데, at most 하나의 field가 set된다면, oneof를 사용해서 memory를 save할 수 있다.

oneof의 활용 예시

```protobuf
message SampleMessage {
  oneof test_oneof {
    string name = 4;
    SubMessage sub_message = 9;
  }
}
```

`test_oneof`가 oneof의 이름에 해당하고, 두 개의 fields가 oneof의 definition에 정의되었다. 여기에는 모든 types of fields가 가능하지만, `map`과 `repeated` fields를 사용될 수 없다.

### maps

data definition에 associative map을 형성할 수 있다. 다음과 같은 shortcut으로 definition을 구현할 수 있다.

```protobuf
map<key_type, value_type> map_field = N;
```

`key_type`은 any integral 또는 string type이 될 수 있으나 enum은 될 수 없다. 그리고 `value_type`으로는 다른 map이 사용될 수 없다.

예시, map of projects where each `Project` message is associated with a string key.

```protobuf
map<string, Project> projects = 3;
```

map field는 `repeated`가 될 수 없음. 그외 map관련 특성:

- Wire format ordering and map iteration ordering of map values are undefined, so you cannot rely on your map items being in a particular order.
- When generating text format for a `.proto`, maps are sorted by key. Numeric keys are sorted numerically.
- When parsing from the wire or when merging, if there are duplicate map keys the last key seen is used. When parsing a map from text format, parsing may fail if there are duplicate keys.
- If you provide a key but no value for a map field, the behavior when the field is serialized is language-dependent. In C++, Java, Kotlin, and Python the default value for the type is serialized, while in other languages nothing is serialized.

### defining services

RPC(Remote Procedure Call) system과 message types를 사용하려면, RPC service interface를 `.proto` 파일에 define해야한다. 

예시, define an RPC service with a method that takes your `SearchRequest` and returns a `SearchResponse`, you can define it in your `.proto` file as follows:

```protobuf
service SearchService {
  rpc Search(SearchRequest) returns (SearchResponse);
}
```

protocol buffers와 함게 사용하기 가장 간단한 RPC system은 gRPC이다. (gRPC: a language- and platform-neutral open source RPC system developed at Google) 만약 gRPC를 사용하지 않는다면, 직접 구현한 RPC implementation과 함께 protocol buffers를 사용할 수도 있다.

<br>

<br>

## JSON Mapping

proto3에서는 canonical encoding in JSON이 가능하다. system들 사이에서 data가 오고가는 것을 더 쉽게 구현할 수 있다고 한다.

JSON-encoded data를 protocol buffer로 parsing할때에 missing value가 있거나 value가 null이면, 이는 corresponding default value로 해석된다.

When generating JSON-encoded output from a protocol buffer, if a protobuf field has the default value and if the field doesn’t support field presence, it will be omitted from the output by default. An implementation may provide options to include fields with default values in the output.

A proto3 field that is defined with the `optional` keyword supports field presence. Fields that have a value set and that support field presence always include the field value in the JSON-encoded output, even if it is the default value.

<br>

<br>

# Python Tutorial (example)

"address book" application에서 사람들의 contact details를 file에서/ file으로 읽고 쓰는 것을 protocol buffer를 통해 구현할 수 있다.

Each person in the address book has :

- a name, 
- an ID, 
- an email address, 
- a contact phone number

<br>

<br>

## Defining proto

가장 먼저 .proto 파일에 serialize할 data structure를 message형태로 define한다.

```protobuf
syntax = "proto3";

// 일반적으로 directory structure를 package이름으로 지정. code 생성에는 직접적인 영향 없음.
package tutorial;

// message = an aggregate containing a set of typed fields
message Person {
  optional string name = 1;
  optional int32 id = 2;
  optional string email = 3;

  enum PhoneType {
    MOBILE = 0;
    HOME = 1;
    WORK = 2;
  }

  message PhoneNumber {
    optional string number = 1;
    optional PhoneType type = 2 [default = HOME];
  }

  repeated PhoneNumber phones = 4;
}

message AddressBook {
  repeated Person people = 1;
}
```

`AddressBook` message안에는 `Person` message가 포함되어있고, `Person` message안에는 `PhoneNumber` message가 포함되어있다. (nested type)

각 field에는 binary encoding에서 field를 식별하기 위해 필요한 unique "tag"가 field number로 달려있다. 위에서는 name은 1, id는 2, email는 3으로 지정되어있다.

각 field는 다음 3가지 modifiers 중 하나가 꼭 명시되어야 한다:

- `optional`: the field may or may not be set. 값이 set되어 있지 않은경우엔 자동으로 default 값이 set된다. 위 예시에서는 for the phone number `type` 으로 default가 설정되어있다. Otherwise, a system default is used: zero for numeric types, the empty string for strings, false for bools. For embedded messages, the default value is always the “default instance” or “prototype” of the message, which has none of its fields set. Calling the accessor to get the value of an optional (or required) field which has not been explicitly set always returns that field’s default value.
- `repeated`: field가 반복될 수 있다. The order of the repeated values will be preserved in the protocol buffer. Think of repeated fields as dynamically sized arrays. 
- `required`: field의 값이 반드시 설정되어야 한다. otherwise the message will be considered “uninitialized”. Serializing an uninitialized message will raise an exception. 에러가 발생할 수 있는 risk가 있는 modifier이기때문에 반드시 필요한것이 아니라면 피하는것이 좋음.

<br>

<br>

## Compiling protocol buffers

```python
protoc -I=$SRC_DIR --python_out=$DST_DIR $SRC_DIR/addressbook.proto
```

- [download the package](https://protobuf.dev/downloads) : compiler를 download 한뒤, 
- compiler를 위 command-line으로 실행한다. specifying the source directory (where your application’s source code lives – the current directory is used if you don’t provide a value), the destination directory (where you want the generated code to go; often the same as `$SRC_DIR`), and the path to your `.proto`.

위 과정을 실행하여 `addressbook_pb2.py` 를 지정한 destination directory안에 생성한다.

<br>

<br>

## Protocol Buffer API

Python protocol buffer compiler는 data access code를 직접 생성하지 않고(C++, Java protocol buffer code와의 다른점), 모든 messages, enums, fields를 위해 다음과 같이 special descriptors를 생성한다.

```python
class Person(message.Message):
  __metaclass__ = reflection.GeneratedProtocolMessageType

  class PhoneNumber(message.Message):
    __metaclass__ = reflection.GeneratedProtocolMessageType
    DESCRIPTOR = _PERSON_PHONENUMBER
  DESCRIPTOR = _PERSON

class AddressBook(message.Message):
  __metaclass__ = reflection.GeneratedProtocolMessageType
  DESCRIPTOR = _ADDRESSBOOK
```

각 class에서 다음 line이 매우 중요하다:

`__metaclass__ = reflection.GeneratedProtocolMessageType`. 

classes를 생성할때에 필요한 template과 같은 역할을 한다.

At load time, the `GeneratedProtocolMessageType` metaclass uses the specified descriptors to create all the Python methods you need to work with each message type and adds them to the relevant classes. You can then use the fully-populated classes in your code.

결국 `Message` base class의 각 field를 regular field로 define한듯 `Person` class를 가져다 사용할 수 있다. 사용할 수 있는 예시는 다음과 같다.

```python
import addressbook_pb2
person = addressbook_pb2.Person()
person.id = 1234
person.name = "John Doe"
person.email = "jdoe@example.com"
phone = person.phones.add()
phone.number = "555-4321"
phone.type = addressbook_pb2.Person.HOME
```

enum으로 define되었던 PhoneType은 set of symbolic constants with integers로 구현된다. So, for example, the constant `addressbook_pb2.Person.PhoneType.WORK` has the value 2  

그리고 다음 standard message methods를 통해 entire message를 manipulate할 수 있다:

- `IsInitialized()`: checks if all the required fields have been set.
- `__str__()`: returns a human-readable representation of the message, particularly useful for debugging. (Usually invoked as `str(message)` or `print message`.)
- `CopyFrom(other_msg)`: overwrites the message with the given message’s values.
- `Clear()`: clears all the elements back to the empty state.

### parsing and serialization

Each protocol buffer class has methods for writing and reading messages of your chosen type using the protocol buffer [binary format](https://protobuf.dev/programming-guides/encoding). These include:

- `SerializeToString()`: serializes the message and returns it as a string. Note that the bytes are binary, not text; we only use the `str` type as a convenient container.
- `ParseFromString(data)`: parses a message from the given string.

<br>

<br>

## Writing a Message

AI server에서 prediction 결과를 Message로 적을때 

생성된 protocol buffer class를 활용해서 address book application이 personal contact details를 address book file에 적는다. 이를 위해 protocol buffer class의 instance를 생성하고 populate한 뒤, write them to an output stream.

- reads an `AddressBook` from a file, 

- adds one new `Person` to it based on user input, and 

- writes the new `AddressBook` back out to the file again. 

위 과정이 다음 code로 구현한다.

```python
#! /usr/bin/python

import addressbook_pb2
import sys

# This function fills in a Person message based on user input.
def PromptForAddress(person):
  person.id = int(raw_input("Enter person ID number: "))
  person.name = raw_input("Enter name: ")

  email = raw_input("Enter email address (blank for none): ")
  if email != "":
    person.email = email

  while True:
    number = raw_input("Enter a phone number (or leave blank to finish): ")
    if number == "":
      break

    phone_number = person.phones.add()
    phone_number.number = number

    type = raw_input("Is this a mobile, home, or work phone? ")
    if type == "mobile":
      phone_number.type = addressbook_pb2.Person.PhoneType.MOBILE
    elif type == "home":
      phone_number.type = addressbook_pb2.Person.PhoneType.HOME
    elif type == "work":
      phone_number.type = addressbook_pb2.Person.PhoneType.WORK
    else:
      print "Unknown phone type; leaving as default value."

# Main procedure:  Reads the entire address book from a file,
#   adds one person based on user input, then writes it back out to the same
#   file.
if len(sys.argv) != 2:
  print "Usage:", sys.argv[0], "ADDRESS_BOOK_FILE"
  sys.exit(-1)

address_book = addressbook_pb2.AddressBook()

# Read the existing address book.
try:
  f = open(sys.argv[1], "rb")
  address_book.ParseFromString(f.read())
  f.close()
except IOError:
  print sys.argv[1] + ": Could not open file.  Creating a new one."

# Add an address.
PromptForAddress(address_book.people.add())

# Write the new address book back to disk.
f = open(sys.argv[1], "wb")
f.write(address_book.SerializeToString())
f.close()
```

<br>

<br>

## Reading a Message

AI server로 input data를 Message로 읽을때

Address book에서 정보를 읽어온다.

```python
#! /usr/bin/python

import addressbook_pb2
import sys

# Iterates though all people in the AddressBook and prints info about them.
def ListPeople(address_book):
  for person in address_book.people:
    print "Person ID:", person.id
    print "  Name:", person.name
    if person.HasField('email'):
      print "  E-mail address:", person.email

    for phone_number in person.phones:
      if phone_number.type == addressbook_pb2.Person.PhoneType.MOBILE:
        print "  Mobile phone #: ",
      elif phone_number.type == addressbook_pb2.Person.PhoneType.HOME:
        print "  Home phone #: ",
      elif phone_number.type == addressbook_pb2.Person.PhoneType.WORK:
        print "  Work phone #: ",
      print phone_number.number

# Main procedure:  Reads the entire address book from a file and prints all
#   the information inside.
if len(sys.argv) != 2:
  print "Usage:", sys.argv[0], "ADDRESS_BOOK_FILE"
  sys.exit(-1)

address_book = addressbook_pb2.AddressBook()

# Read the existing address book.
f = open(sys.argv[1], "rb")
address_book.ParseFromString(f.read())
f.close()

ListPeople(address_book)
```

<br>

<br>

## Python Generated Code Guide

### compiler invocation

`--python_out=` command-line flag를 통해 protocol buffer compiler가 Python output을 생성한다.

```python
# example:
protoc --proto_path=src --python_out=build/gen src/foo.proto src/bar/baz.proto
```

The compiler will read the files `src/foo.proto` and `src/bar/baz.proto` and produce two output files: `build/gen/foo_pb2.py` and `build/gen/bar/baz_pb2.py`. The compiler will automatically create the directory `build/gen/bar` if necessary, but it will *not* create `build` or `build/gen`; they must already exist.

<br>

### Messages

protocol buffer copmiler가 message로 declare된 것을 class로 생성한다.

```protobuf
message Foo{}
```

위와 같이 declare된 message는 Foo class로 생성된다. 이는 `google.protobuf.Message.`의 subclass이다.

You should *not* create your own `Foo` subclasses. Generated classes are not designed for subclassing and may lead to "fragile base class" problems. Besides, implementation inheritance is bad design.

#### nested types

message안에 다른 message가 declare될 수 있다.

```protobuf
message Foo {
  message Bar {}
}
```

n this case, the `Bar` class is declared as a static member of `Foo`, so you can refer to it as `Foo.Bar`

<br>

### Well known Types

messages외에 다른 well known types를 proto file에 설정하여 활용할 수 있다. 이들은 `google.protobuf.Message`](https://googleapis.dev/python/protobuf/latest/google/protobuf/message.html#google.protobuf.message.Message) and a WKT class의 subclass에 해당한다.

Well known types: Any, Timestamp, Duration, FieldMask, Struct, ListValue

<Br>

### Fields

Message type안에 각 field는 message가 해당하는 Python class의 property와 동등하다고 생각하면 된다. 그리고 compiler가 각 field에 integer constant를 field number로 지정해준다. For example, given the field `optional int32 foo_bar = 5;`, the compiler will generate the integer constant `FOO_BAR_FIELD_NUMBER = 5`.

#### singular fields

#### singular message fields

#### repeated fields

repeated fields는 Python sequence와 비슷한 object라고 생각하면 된다. 

```protobuf
message Foo {
  repeated int32 nums = 1;
}
```

#### repeated message fields

repeated message fields는 repeated scalar fields와 비슷하게 동작한다.

```protobuf
message Foo {
  repeated Bar bars = 1;
}
message Bar {
  optional int32 i = 1;
  optional int32 j = 2;
}
```

#### map fields

map fields는 Python dict와 동등하다.

다음과 같이 message definition이 주어진다면:

```protobuf
message MyMessage {
  map<int32, int32> mapfield = 1;
}
```

Map fields를 위해 다음과 같이 Python API를 구현 할 수 있다:

```python
# Assign value to map
m.mapfield[5] = 10

# Read value from map
m.mapfield[5]

# Iterate over map keys
for key in m.mapfield:
  print(key)
  print(m.mapfield[key])

# Test whether key is in map:
if 5 in m.mapfield:
  print(“Found!”)

# Delete key from map.
del m.mapfield[key]
```

<br>

### enumerations

In Python, enums are just integers. A set of integral constants are defined corresponding to the enum’s defined values. For example, given:

```protobuf
message Foo {
  enum SomeEnum {
    VALUE_A = 0;
    VALUE_B = 5;
    VALUE_C = 1234;
  }
  optional SomeEnum bar = 1;
}
```

<br>

### Services

#### Interface

service definition이 다음과 같이 interface에서 정의되면,

```protobuf
service Foo {
  rpc Bar(FooRequest) returns(FooResponse);
}
```

The protocol buffer compiler will generate a class `Foo` to represent this service. `Foo` will have a method for each method defined in the service definition. In this case, the method `Bar` is defined as:

```python
def Bar(self, rpc_controller, request, done)
```

#### Stub

The protocol buffer compiler also generates a "stub" implementation of every service interface, which is used by clients wishing to send requests to servers implementing the service. For the `Foo` service (above), the stub implementation `Foo_Stub` will be defined.

`Foo_Stub` is a subclass of `Foo`. Its constructor takes an [`RpcChannel`](https://googleapis.dev/python/protobuf/latest/google/protobuf/service.html#google.protobuf.service.RpcChannel) as a parameter. The stub then implements each of the service’s methods by calling the channel’s `CallMethod()` method.

<br>

<br>

# 주의할 점

data에 null이 있는경우 encoding - parsing 과정에서 원치않는 data 변형이 일어나지는 않는지 확인이 필요하다. null에 해당하는 [default value][https://protobuf.dev/programming-guides/proto3/#default]의 확인이 필요함. (JSON Mapping 참고)

## repeated fields vs. stream service

```protobuf
syntax = "proto3";

import "google/protobuf/empty.proto";

message Dummy {
  string foo = 1;
  string bar = 2;
}

message DummyList {
  repeated Dummy dummy = 1;
}

service DummyService {
  rpc getDummyListWithStream(google.protobuf.Empty) returns (stream Dummy) {}
  rpc getDummyListWithRepeated(google.protobuf.Empty) returns (DummyList) {}
}
```

Microsoft에서 제공하는 tutorial에서는 다음과 같은 요소들을 고려해서 결정하라고 제안한다.

- The overall size of the dataset.
- The time it took to create the dataset at either the client or server end.
- Whether the consumer of the dataset can start acting on it as soon as the first item is available, or needs the complete dataset to do anything useful.

**repeated fields 사용 case**

> 첫 번째 message를 define하고, 두 번째 message를 define해서 첫 번째 message가 repeated field로 정의된다. Declare a list or arrays of messages within another message. 그리고 service method를 하나 정의하여 두 번째 message를 return하도록 한다. 

dataset의 size가 제한적이고, set 전체가 짧은 시간 내에 생성될 수 있는 경우 (i.e, under 1 sec), 또는 "batching"의 성격을 띄고있는 경우 repeated field를 사용하는 것이 적합하다. dataset을 보내는 쪽에서 set전체를 보내기 전에 다 준비해야하고, 받는 쪽에서도 set을 다 받아야 데이터의 처리가 시작될 수 있는 경우이다. 또한, repeated fields가 highly compressible하다면, single message로 전송하는 것이 더 효율적이다.

e.g., e-commerce system, to build a list of items within an order (assuming that the list won't be very large)

<br>

**stream service 사용 case**

> message를 하나 define하고, 이 message를 stream 형태로 return하는 service method를 정의한다. Utilize a long-running persistent connection

dataset의 size가 크고, message를 받는쪽에서 incoming message가 도달하는 대로 바로바로 처리 가능하다면, stream이 더 적합하다. construct a large object in memory, write it to  the network, then free up the resources. service의 scalability를 개선할 수 있는 자율성이 더 주어진다. 받는쪽에서 모든 incoming messages가 도착하기까지 blocking해야 하는 경우 repeated fields 방식이 더 적합하겠지만, 이런 case에서도 stream방식도 적절하다는 의견도 있다[5].

<br>

<br>

# References 

1. protobuf language guide https://protobuf.dev/programming-guides/proto3/
2. Protobuf tutorial for Python implementation https://protobuf.dev/reference/python/python-generated/

3. Protocol Buffer Basics: Python https://protobuf.dev/getting-started/pythontutorial/

4. gRPC for WCF Developers https://learn.microsoft.com/ko-kr/dotnet/architecture/grpc-for-wcf-developers/protobuf-messages

5. https://groups.google.com/g/grpc-io/c/F23vXwilTq0
